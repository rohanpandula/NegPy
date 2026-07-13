from typing import Optional, Any, Callable, Tuple
from negpy.domain.types import ImageBuffer
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.kernel.caching.manager import PipelineCache
from negpy.kernel.caching.logic import calculate_config_hash, CacheEntry
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.image.logic import working_oetf_encode
from negpy.kernel.system.logging import get_logger
from negpy.features.geometry.processor import GeometryProcessor, CropProcessor
from negpy.features.exposure import models as exposure_models
from negpy.features.exposure.models import RenderIntent
from negpy.features.exposure.processor import (
    NormalizationProcessor,
    PhotometricProcessor,
)
from negpy.features.toning.processor import ToningProcessor
from negpy.features.lab.logic import apply_clahe
from negpy.features.lab.processor import PhotoLabProcessor
from negpy.features.retouch.processor import RetouchProcessor
from negpy.features.finish.processor import FinishProcessor
from negpy.features.nikonlook.processor import NikonLookProcessor
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.view.coordinate_mapping import CoordinateMapping

logger = get_logger(__name__)


class DarkroomEngine:
    """
    Runs the pipeline. Handles stage caching.
    """

    def __init__(self) -> None:
        self.config = APP_CONFIG
        self.cache = PipelineCache()

    def _run_stage(
        self,
        img: ImageBuffer,
        config: Any,
        cache_field: str,
        processor_fn: Callable[[ImageBuffer, PipelineContext], ImageBuffer],
        context: PipelineContext,
        pipeline_changed: bool,
    ) -> Tuple[ImageBuffer, bool]:
        conf_hash = calculate_config_hash(config)
        cached_entry = getattr(self.cache, cache_field)

        if not pipeline_changed and cached_entry and cached_entry.config_hash == conf_hash:
            context.metrics.update(cached_entry.metrics)
            context.active_roi = cached_entry.active_roi
            return cached_entry.data, False

        new_img = processor_fn(img, context)
        new_entry = CacheEntry(conf_hash, new_img, context.metrics.copy(), context.active_roi)
        setattr(self.cache, cache_field, new_entry)

        return new_img, True

    def process(
        self,
        img: ImageBuffer,
        settings: WorkspaceConfig,
        source_hash: str,
        context: Optional[PipelineContext] = None,
    ) -> ImageBuffer:
        img = ensure_image(img)
        h_orig, w_cols = img.shape[:2]

        if context is None:
            context = PipelineContext(
                scale_factor=max(h_orig, w_cols) / float(self.config.preview_render_size),
                original_size=(h_orig, w_cols),
                process_mode=settings.process.process_mode,
            )

        pipeline_changed = False
        if self.cache.source_hash != source_hash:
            self.cache.clear()
            self.cache.source_hash = source_hash
            pipeline_changed = True

        if self.cache.process_mode != settings.process.process_mode:
            self.cache.process_mode = settings.process.process_mode
            self.cache.base = None
            self.cache.exposure = None
            self.cache.clahe = None
            self.cache.retouch = None
            self.cache.lab = None
            pipeline_changed = True

        current_img = img

        if settings.geometry.manual_crop_rect:
            logger.debug(f"Engine process with manual_crop_rect: {settings.geometry.manual_crop_rect}")

        # Folded into the base stage like fine_rotation (not source_hash) so the slider
        # re-renders without re-decoding the RAW.
        distortion_k1 = settings.flatfield.k1 if settings.flatfield.apply else 0.0

        def run_base(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
            img_in = GeometryProcessor(settings.geometry, distortion_k1).process(img_in, ctx)
            return NormalizationProcessor(settings.process).process(img_in, ctx)

        # While the crop tool shows the full uncropped frame, the crop-selection
        # fields (manual_crop_rect, auto_crop_*) only feed context.active_roi, which
        # is itself unused for output in that mode (CropProcessor and uv_grid ROI
        # slicing are both bypassed) - keying on them would force a full base/
        # exposure/retouch/lab/local recompute on every crop-rect drag step.
        geometry_key = (
            (
                settings.geometry.rotation,
                settings.geometry.fine_rotation,
                settings.geometry.flip_horizontal,
                settings.geometry.flip_vertical,
            )
            if context.crop_preview_full
            else settings.geometry
        )

        base_key = (
            settings.process.process_mode,
            settings.process.e6_normalize,
            geometry_key,
            settings.process.analysis_buffer,
            settings.process.analysis_rect,
            settings.process.luma_range_clip,
            settings.process.color_range_clip,
            settings.process.use_luma_average,
            settings.process.use_colour_average,
            settings.process.is_local_initialized,
            settings.process.is_locked_initialized,
            settings.process.locked_floors,
            settings.process.locked_ceils,
            settings.process.local_floors,
            settings.process.local_ceils,
            settings.process.white_point_offset,
            settings.process.black_point_offset,
            settings.process.white_point_trim_red,
            settings.process.white_point_trim_green,
            settings.process.white_point_trim_blue,
            settings.process.black_point_trim_red,
            settings.process.black_point_trim_green,
            settings.process.black_point_trim_blue,
            settings.process.crosstalk_strength,
            settings.process.crosstalk_matrix,
            settings.process.lock_bounds,
            distortion_k1,
            # Auto Density metering reads retuned targets from EXPOSURE_CONSTANTS,
            # which no config hash sees; the revision keys them in. Re-running base
            # sets pipeline_changed, so the exposure stage follows.
            exposure_models.TARGETS_REVISION,
        )
        current_img, pipeline_changed = self._run_stage(current_img, base_key, "base", run_base, context, pipeline_changed)

        def run_exposure(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
            img_out = PhotometricProcessor(settings.exposure, settings.local).process(img_in, ctx)
            return img_out

        # Dodge/burn masks are print-exposure inputs, so they key this stage.
        current_img, pipeline_changed = self._run_stage(
            current_img,
            (settings.exposure, settings.local),
            "exposure",
            run_exposure,
            context,
            pipeline_changed,
        )

        # Flat (digital-intermediate) master: keep only geometry + mask-neutralized
        # inversion, then crop. All creative stages (retouch, lab, local, toning,
        # finish) are bypassed so the export holds maximal editing latitude.
        flat_intent = settings.exposure.render_intent == RenderIntent.FLAT

        if not flat_intent:

            def run_clahe(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
                return apply_clahe(img_in, settings.lab.clahe_strength)

            # Keyed on the bare float: the full settings.lab would wrongly invalidate
            # this stage (and retouch/lab behind it) on saturation/sharpen edits.
            current_img, pipeline_changed = self._run_stage(
                current_img,
                settings.lab.clahe_strength,
                "clahe",
                run_clahe,
                context,
                pipeline_changed,
            )

            def run_retouch(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
                return RetouchProcessor(settings.retouch).process(img_in, ctx)

            current_img, pipeline_changed = self._run_stage(
                current_img,
                settings.retouch,
                "retouch",
                run_retouch,
                context,
                pipeline_changed,
            )

            def run_lab(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
                return PhotoLabProcessor(settings.lab).process(img_in, ctx)

            current_img, pipeline_changed = self._run_stage(current_img, settings.lab, "lab", run_lab, context, pipeline_changed)

            current_img = ToningProcessor(settings.toning).process(current_img, context)

        if not context.crop_preview_full:
            current_img = CropProcessor(settings.geometry).process(current_img, context)

        if not flat_intent:
            from negpy.services.export.print import PrintService

            paper = PrintService.effective_paper_linear(settings.finish, settings.toning)
            current_img = FinishProcessor(settings.finish, settings.export.export_print_size, paper).process(current_img, context)
            # Output transform: scene-linear -> display-encoded (flat master skips this).
            current_img = ensure_image(working_oetf_encode(current_img))

        if context.wants_uv_grid:
            try:
                uv_grid = CoordinateMapping.create_uv_grid(
                    rh_orig=h_orig,
                    rw_orig=w_cols,
                    rotation=settings.geometry.rotation,
                    fine_rot=settings.geometry.fine_rotation,
                    flip_h=settings.geometry.flip_horizontal,
                    flip_v=settings.geometry.flip_vertical,
                    autocrop=True,
                    autocrop_params=({"roi": context.active_roi} if context.active_roi and not context.crop_preview_full else None),
                    distortion_k1=distortion_k1,
                )
                context.metrics["uv_grid"] = uv_grid
            except Exception as e:
                logger.error(f"Failed to generate UV grid: {e}")

        return current_img

    def process_nikonlook(
        self,
        img: ImageBuffer,
        settings: WorkspaceConfig,
        context: PipelineContext,
    ) -> ImageBuffer:
        """
        "Nikon Scan look (beta)" -- an ADDITIVE, entirely separate render
        path from `.process()` above (never touches its `_run_stage`/
        `PipelineCache` machinery or control flow; `.process()`'s own body
        is untouched by this feature). See
        digital-ice-2026/negfit/NEGPY-INTEGRATION-PLAN.md for the design.

        Runs geometry (shared with `.process()`, unmodified) and then hands
        off entirely to `NikonLookProcessor`: NegPy's own normalization
        (base stage) and print-curve (exposure stage) are REPLACED by the
        external negfit color model, not composed with them -- the
        bundle's Layer B already produces a complete, final look, so the
        normal retouch/lab/toning/finish creative stages and the working-
        space OETF encode are bypassed too (mirrors how `flat_intent`
        bypasses those same stages above, for the same reason: the output
        is already a finished artifact, not a scene-linear intermediate
        meant for further stage math). Crop still applies -- a framing
        correction, orthogonal to color science.

        Caller contract: `settings.nikonlook.nikonlook_enabled` must be True (checked
        by `ImageProcessor._is_nikonlook`, not re-checked here) and
        `settings.process.linear_raw` must be True -- this mode needs raw,
        NOT white-balanced sensor RGB (the bundle's matrix already performs
        the equivalent correction; see nikonlook_core's module docstring).
        Raises ValueError if linear_raw is False rather than silently
        producing wrong colors.
        """
        if not settings.process.linear_raw:
            raise ValueError(
                "Nikon Scan look (beta) requires settings.process.linear_raw=True "
                "(the color model expects raw, non-white-balanced sensor RGB -- see "
                "digital-ice-2026/negfit/nikonlook_core.py's module docstring). "
                "Enable Linear Raw before using this mode."
            )

        img = ensure_image(img)
        distortion_k1 = settings.flatfield.k1 if settings.flatfield.apply else 0.0
        img = GeometryProcessor(settings.geometry, distortion_k1).process(img, context)
        ir_post_geometry = context.metrics.get("ir_post_geometry")

        device_rgb01, icc_bytes = NikonLookProcessor(settings.nikonlook).process(
            img,
            ir_post_geometry,
            settings.retouch,
            context.scale_factor,
        )
        # Side-channel back to ImageProcessor's export encoder -- see
        # image_processor.py's `_encode_export_nikonlook`, which tags
        # these bytes directly instead of running normal ICC color
        # management (this buffer is already final device values in the
        # bundle's own ICC space, per NikonLookProcessor.process's contract).
        context.metrics["nikonlook_icc_bytes"] = icc_bytes

        if not context.crop_preview_full:
            device_rgb01 = CropProcessor(settings.geometry).process(device_rgb01, context)

        return ensure_image(device_rgb01)

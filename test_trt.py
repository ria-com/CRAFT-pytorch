"""
Copyright (c) 2019-present NAVER Corp.
MIT License

TensorRT 10+ Adaptation by [Your Name/Handle]
"""

# tensorrt_infer.py
# Працює з TensorRT 10.9.0.34  (pip install nvidia-pyindex tensorrt==10.9.0.post12 \
#                                cuda-python pycuda)
import os
from typing import Dict, Any, List, Tuple
import imgproc
import numpy as np
import tensorrt as trt
import cuda  # from cuda-python pkg
import pycuda.driver as cuda_drv
import pycuda.autoinit           # одноразова ініціалізація CUDA-контексту
import craft_utils
import file_utils

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def _dtype_of(tensor_dtype: trt.DataType):
    """TensorRT dtype → NumPy dtype."""
    return {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int32:   np.int32,
        trt.int8:    np.int8,
        trt.bool:    np.bool_,
    }[tensor_dtype]


class TrtRunner:
    """
    Загальний раннер для будь-якого TensorRT-двигуна (.trt).

    ● Підтримує dynamic shape та декілька входів/виходів.
    ● Розподіляє pinned-host та device-пам’ять, кешує її за формою.
    ● API:
        >>> runner = TrtRunner("model.trt")
        >>> outputs = runner(inputs_dict)          # dict name→np.ndarray
    """

    def __init__(self, engine_path: str, device_id: int = 0):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        self.stream = cuda_drv.Stream()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(f.read())
        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        self.device_id = device_id

        # I/O-метадані
        self.input_names: List[str] = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        ]
        self.output_names: List[str] = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]

        # Кеш буферів keyed by shape tuple
        self._buffer_cache: Dict[Tuple[Tuple[int, ...], ...], Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _allocate_buffers(self, input_shapes: Dict[str, Tuple[int, ...]]):
        """
        Внутрішнє: виділяє pinned-host та device-пам’ять під конкретну
        комбінацію форм. Результат кешується, щоб не робити malloc на кожен батч.
        """
        cache_key = tuple(sorted(input_shapes.items()))
        if cache_key in self._buffer_cache:
            return self._buffer_cache[cache_key]

        bindings: Dict[str, Dict[str, Any]] = {}
        for name in self.engine:
            mode = self.engine.get_tensor_mode(name)
            dtype = _dtype_of(self.engine.get_tensor_dtype(name))
            shape = (
                input_shapes[name]
                if mode == trt.TensorIOMode.INPUT
                else self.engine.get_tensor_shape(name)
            )

            # Для output із dynamic dim залежних від input батчу ―
            # форму уточнимо вже після set_input_shape + resolve.
            vol = int(np.prod(shape)) if -1 not in shape else 0
            host_mem = (
                cuda_drv.pagelocked_empty(vol, dtype) if vol > 0 else None
            )  # виділимо пізніше
            dev_mem = (
                cuda_drv.mem_alloc(host_mem.nbytes) if host_mem is not None else None
            )

            bindings[name] = {
                "host": host_mem,
                "device": dev_mem,
                "dtype": dtype,
                "shape": shape,
                "mode": mode,
            }

        # збережемо
        self._buffer_cache[cache_key] = bindings
        return bindings

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Запуск інференсу.

        Args:
            inputs: dict {tensor_name: np.ndarray} ― форми/типи мають відповідати engine.

        Returns:
            dict {output_name: np.ndarray}
        """
        # ---- 1. прописуємо dynamic shapes ----
        for name, arr in inputs.items():
            self.context.set_input_shape(name, tuple(arr.shape))
        assert self.context.all_binding_shapes_specified

        # ---- 2. готуємо (або відновлюємо з кешу) буфери ----
        input_shapes = {n: tuple(arr.shape) for n, arr in inputs.items()}
        bindings = self._allocate_buffers(input_shapes)

        # ---- 3. встановлюємо адреси тензорів ----
        for name, buf in bindings.items():
            if buf["device"] is None:  # output, форму тепер знаємо
                shape = tuple(self.context.get_tensor_shape(name))
                vol = int(np.prod(shape))
                host_mem = cuda_drv.pagelocked_empty(vol, buf["dtype"])
                dev_mem = cuda_drv.mem_alloc(host_mem.nbytes)
                buf["host"], buf["device"], buf["shape"] = host_mem, dev_mem, shape
            self.context.set_tensor_address(name, int(buf["device"]))

        # ---- 4. копіюємо входи host→device ----
        for name, arr in inputs.items():
            buf = bindings[name]
            np.copyto(buf["host"].reshape(arr.shape), arr)
            cuda_drv.memcpy_htod_async(buf["device"], buf["host"], self.stream)

        # ---- 5. inference ----
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # ---- 6. device→host для виходів ----
        outputs: Dict[str, np.ndarray] = {}
        for name in self.output_names:
            buf = bindings[name]
            cuda_drv.memcpy_dtoh_async(buf["host"], buf["device"], self.stream)
        self.stream.synchronize()

        for name in self.output_names:
            buf = bindings[name]
            outputs[name] = buf["host"].reshape(buf["shape"]).copy()  # detach

        return outputs


# ---------------------------------------------------------------------- #
# Спеціалізовані обгортки під ваші CRAFT та RefineNet
# ---------------------------------------------------------------------- #
class CraftTrtInference:
    """
    Зручна обгортка: бере BGR/RGB float32|float16 з нормалізацією в
    [0,1], (N,3,H,W) → повертає ('regions','affinity').
    """

    def __init__(self, engine_path: str):
        self.runner = TrtRunner(engine_path)

    def __call__(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if images.ndim != 4:
            raise ValueError("Очікую тензор (N,3,H,W)")
        #if images.dtype not in (np.float32, np.float16):
        #    images = images.astype(np.float32) / 255.0
        outputs = self.runner({"images": images})
        return outputs["regions"], outputs["affinity"]


class RefineNetTrtInference:
    """
    Вхід = два тензори:
        craft_y   ― (N, H/2, W/2, 2)  float32|float16
        features  ― (N, 32,  H/2, W/2)
    Повертає refined_affinity.
    """

    def __init__(self, engine_path: str):
        self.runner = TrtRunner(engine_path)

    def __call__(
        self, craft_y: np.ndarray, craft_features: np.ndarray
    ) -> np.ndarray:
        outputs = self.runner(
            {
                "craft_output_y": craft_y.astype(np.float16)
                if craft_y.dtype == np.float32
                else craft_y,
                "craft_output_features": craft_features.astype(np.float16)
                if craft_features.dtype == np.float32
                else craft_features,
            }
        )
        return outputs["refined_affinity"]


# ---------------------------------------------------------------------- #
# Приклад використання
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import cv2, time
    craft_engine = "/mnt/raid/var/www/modelhub-client-trt/data/models/craft/craft_mlt_25k_2020-02-16.trt"
    refine_engine = "/mnt/raid/var/www/modelhub-client-trt/data/models/craft/refinenet_fixed.trt"

    # 1. Cinit
    craft_inf = CraftTrtInference(craft_engine)
    refine_inf = RefineNetTrtInference(refine_engine)

    t = time.time()
    # image prepere
    image_path = "./images/sample.jpg"
    image = imgproc.loadImage(image_path)

    # resize and normalize image
    canvas_size = 1024
    mag_ratio = 1.5
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

    # predict
    t0 = time.time()
    reg, aff = craft_inf(x)
    score_text = reg[0, :, :, 0]
    score_link = reg[0, :, :, 1]

    t0 = time.time()
    y_refiner = refine_inf(reg, aff)
    score_link = y_refiner[0, :, :, 0]

    low_text = 0.4
    link_threshold = 0.4
    text_threshold = 0.7
    poly = False
    score_text = score_text.astype(np.float32)
    score_link = score_link.astype(np.float32)

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # save score text
    result_folder = './result/'
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/trt_res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, ret_score_text)

    file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))

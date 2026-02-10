from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fastapi import UploadFile
from loguru import logger
from skimage.feature import canny
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and convert it to RGB.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        RGB image with shape (H, W, 3).
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_image(
    image: np.ndarray,
    *,
    title: str = None,
    figsize: tuple = (8, 6),
):
    """
    Display an RGB image using matplotlib.

    Parameters
    ----------
    image : np.ndarray
        RGB image with shape (H, W, 3).

    title : str
        Title of the matplotlib figure.

    figsize : tuple
        Figure size passed to matplotlib, e.g. (8, 6).
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def plot_image_with_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    *,
    title: str = "",
    line_color: tuple = (0, 255, 0),
    line_thickness: int = 1,
    figsize: tuple = (8, 6),
):
    """
    Display an image with a single polygon overlay.

    Parameters
    ----------
    image : np.ndarray
        RGB image with shape (H, W, 3).

    polygon : np.ndarray
        Polygon coordinates with shape (N, 2), N >= 3.
        Coordinates must be ordered clockwise or counter-clockwise.
        Example:
            np.array([
                [x1, y1],
                [x2, y2],
                [x3, y3],
                [x4, y4],
            ], dtype=np.int32)

    title : str
        Title of the matplotlib figure.

    line_color : tuple
        Polygon color in RGB format.

    line_thickness : int
        Thickness of polygon edges.

    figsize : tuple
        Figure size passed to matplotlib.
    """
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must have shape (N, 2)")

    img_out = image.copy()

    cv2.polylines(
        img_out,
        [polygon.astype(np.int32)],
        isClosed=True,
        color=line_color,
        thickness=line_thickness,
    )

    plot_image(
        img_out,
        title=title,
        figsize=figsize,
    )


def warp_polygon_to_square(
    image: np.ndarray,
    polygon: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """
    Warp a 4-point tilted polygon to a square image.

    Parameters
    ----------
    image : np.ndarray
        RGB input image.

    polygon : np.ndarray
        Fireplace polygon of shape (4, 2), ordered clockwise or counter-clockwise.

    output_size : int
        Target square size in pixels.

    Returns
    -------
    np.ndarray
        Warped square RGB image.
    """
    if polygon.shape != (4, 2):
        raise ValueError("polygon must have shape (4, 2)")

    src_pts = polygon.astype(np.float32)

    dst_pts = np.float32(
        [
            [0, 0],
            [output_size, 0],
            [output_size, output_size],
            [0, output_size],
        ]
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (output_size, output_size))


def split_into_horizontal_bands(
    image: np.ndarray,
    n_bands: int,
) -> List[np.ndarray]:
    """
    Split a square image into horizontal bands.
    """
    h = image.shape[0]
    band_height = h // n_bands

    return [image[i * band_height : (i + 1) * band_height, :] for i in range(n_bands)]


def extract_band_features(
    band: np.ndarray,
    hot_pixel_v_threshold: int,
) -> List[float]:
    """
    Extract visual features from one image band.
    """
    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)

    band_float = band.astype(np.float32)
    rgb_sum = np.sum(band_float, axis=2) + 1e-6
    red_ratio = np.mean(band_float[:, :, 0] / rgb_sum)

    mean_v = np.mean(v)
    std_v = np.std(v)
    mean_s = np.mean(s)

    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    edges = canny(gray / 255.0)
    edge_density = np.mean(edges)

    hot_pixel_ratio = np.mean(v > hot_pixel_v_threshold)

    return [
        mean_v,
        std_v,
        mean_s,
        red_ratio,
        edge_density,
        hot_pixel_ratio,
    ]


def is_black_frame(
    gray: np.ndarray,
    mean_threshold: float,
    black_pixel_threshold: int,
    black_pixel_ratio: float,
) -> bool:
    """
    Determine whether a grayscale image is effectively black.

    This is used to short-circuit prediction at night when the fireplace
    is not visible.

    Args:
        gray: Grayscale image (H x W).
        mean_threshold: Mean pixel value below which the frame is considered black.
        black_pixel_threshold: Pixel value threshold to consider a pixel "black".
        black_pixel_ratio: Fraction of pixels below black_pixel_threshold required
            to classify the frame as black.

    Returns:
        True if the frame is considered black, False otherwise.
    """
    if gray.mean() < mean_threshold:
        return True

    ratio = np.mean(gray < black_pixel_threshold)
    return ratio > black_pixel_ratio


def init_reference(
    gray: np.ndarray,
    orb: cv2.ORB,
) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Initialize ORB reference keypoints and descriptors from a grayscale image.

    Args:
        gray: Grayscale reference image.
        orb: Initialized OpenCV ORB detector.

    Returns:
        A tuple of:
            - ref_gray: Reference grayscale image
            - ref_kp: List of reference keypoints
            - ref_des: Reference descriptors
    """
    kp, des = orb.detectAndCompute(gray, None)
    return gray, kp, des


def update_polygon_with_orb(
    gray: np.ndarray,
    last_polygon: np.ndarray,
    orb: cv2.ORB,
    ref_kp: list,
    ref_des: np.ndarray,
    min_keypoints: int,
    min_matches: int,
    ransac_reproj_threshold: float,
    min_inliers: int,
    max_translation_px: float,
    smoothing_alpha: float | None,
) -> tuple[np.ndarray, bool]:
    """
    Update a polygon location using ORB feature matching and RANSAC.

    Args:
        gray: Current grayscale frame.
        last_polygon: Current polygon (Nx2) in pixel space.
        orb: Initialized ORB detector.
        ref_kp: Reference keypoints.
        ref_des: Reference descriptors.
        min_keypoints: Minimum number of detected keypoints required.
        min_matches: Minimum number of ORB matches required.
        ransac_reproj_threshold: RANSAC reprojection threshold.
        min_inliers: Minimum number of inliers required after RANSAC.
        max_translation_px: Maximum allowed translation magnitude.
        smoothing_alpha: EMA smoothing factor for polygon update,
            or None to disable smoothing.

    Returns:
        (updated_polygon, success)
    """
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < min_keypoints:
        return last_polygon, False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_des, des)

    if len(matches) < min_matches:
        return last_polygon, False

    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])

    M, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
    )

    if M is None or inliers is None:
        return last_polygon, False

    if int(inliers.sum()) < min_inliers:
        return last_polygon, False

    dx, dy = M[0, 2], M[1, 2]
    if np.hypot(dx, dy) > max_translation_px:
        return last_polygon, False

    new_poly = cv2.transform(last_polygon[None, :, :], M)[0]

    if smoothing_alpha is not None:
        updated_poly = smoothing_alpha * last_polygon + (1 - smoothing_alpha) * new_poly
    else:
        updated_poly = new_poly

    return updated_poly, True


def extract_features(
    image: np.ndarray,
    n_bands: int,
    hot_pixel_v_threshold: int,
) -> np.ndarray:
    """
    Extract handcrafted features from horizontal bands of an image.

    Args:
        image: Input BGR image.
        n_bands: Number of horizontal bands.
        hot_pixel_v_threshold: V-channel threshold for hot pixel detection.

    Returns:
        Feature vector as a 1D float32 numpy array.
    """
    bands = split_into_horizontal_bands(image, n_bands)

    features = []
    for band in bands:
        features.extend(extract_band_features(band, hot_pixel_v_threshold))

    return np.array(features, dtype=np.float32)


def load_image_from_upload(upload: UploadFile) -> np.ndarray | None:
    """
    Decode an uploaded image file into a BGR OpenCV image.

    Args:
        upload: FastAPI UploadFile object.

    Returns:
        Decoded BGR image, or None if decoding fails.
    """
    data = np.frombuffer(upload.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def capture_frame_from_stream(stream_url: str) -> np.ndarray:
    """
    Capture a single frame from a video stream.

    Args:
        stream_url: URL of the video stream.

    Returns:
        frame: Captured BGR image as a NumPy array.

    Raises:
        RuntimeError: If the frame cannot be grabbed.
    """
    cap = cv2.VideoCapture(stream_url)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to grab frame from stream")

    return frame


def save_frame_to_path(frame: np.ndarray, store_frame_path: str) -> None:
    """
    Save a BGR image to disk and log the result.

    Args:
        frame: BGR image as a NumPy array.
        store_frame_path: Path where the frame will be saved.

    Raises:
        IOError: If cv2.imwrite fails.
    """
    success = cv2.imwrite(store_frame_path, frame)
    if not success:
        raise IOError(f"Failed to write image to {store_frame_path}")

    logger.debug(f"{datetime.now()} : stored frame to {store_frame_path}")


def capture_image_from_stream(stream_url: str, store_frame_path: str) -> None:
    """
    Capture a frame from a stream and save it to disk.
    Combines capture_frame_from_stream + save_frame_to_path.

    Args:
        stream_url: URL of the video stream.
        store_frame_path: Path where the frame will be saved.
    """
    frame = capture_frame_from_stream(stream_url)
    save_frame_to_path(frame, store_frame_path)
    logger.debug(f"image stored: {store_frame_path}")


def init_fireplace_state(fireplace_poly_init) -> dict:
    return {
        "last_polygon": fireplace_poly_init,
        "ref_gray": None,
        "ref_kp": None,
        "ref_des": None,
    }


def predict_from_image(
    img: np.ndarray,
    clf,
    # orb,
    last_polygon: np.ndarray,
    ref_gray: Optional[np.ndarray],
    ref_kp: Optional[list],
    ref_des: Optional[np.ndarray],
    black_mean_threshold: float,
    black_pixel_threshold: int,
    black_pixel_ratio: float,
    min_keypoints: int,
    min_matches: int,
    ransac_reproj_threshold: float,
    min_inliers: int,
    max_translation_px: int,
    poly_smoothing_alpha: float,
    output_size: Tuple[int, int],
    n_bands: int,
    hot_pixel_v_threshold: float,
    warp_file_path:Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if is_black_frame(
        gray,
        mean_threshold=black_mean_threshold,
        black_pixel_threshold=black_pixel_threshold,
        black_pixel_ratio=black_pixel_ratio,
    ):
        return (
            {
                "label": "black",
                "confidence": 1.0,
                "orb_updated": False,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "last_polygon": last_polygon,
                "ref_gray": ref_gray,
                "ref_kp": ref_kp,
                "ref_des": ref_des,
            },
        )


    warped = warp_polygon_to_square(img, last_polygon, output_size)
    save_frame_to_path(warped, warp_file_path)

    features = extract_features(
        warped,
        n_bands=n_bands,
        hot_pixel_v_threshold=hot_pixel_v_threshold,
    ).reshape(1, -1)

    pred = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]

    return (
        {
            "label": pred,
            "confidence": float(np.max(proba)),
            "orb_updated": False,
            "timestamp": datetime.now().isoformat(),
        },
        {
            "last_polygon": last_polygon,
            "ref_gray": ref_gray,
            "ref_kp": ref_kp,
            "ref_des": ref_des,
        },
    )

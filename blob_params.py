blob_params = {
    "thresholdStep": 10,
    "minThreshold": 190,
    "maxThreshold": 256,
    "minRepeatability": 1,  # to find all blobs,
    "minDistBetweenBlobs": 0,  # pixels,
    "filterByColor": False,  # BROKEN!!!
    "blobColor": 255,
    "filterByArea": True,
    "minArea": 256 * 256 // (256 * 2),  # more than a 256th of the image
    "maxArea": 256 * 256 // 8,  # less than a 16th of the image
    "filterByCircularity": False,  # from 0 to 1
    "minCircularity": 0,
    "maxCircularity": 1,
    "filterByInertia": True,  # from 0 to 1
    "minInertiaRatio": 0,
    "maxInertiaRatio": 1e37,
    "filterByConvexity": True,
    "minConvexity": 0.95,
    "maxConvexity": 1e37,
}

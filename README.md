CombinedVisual.py – Full Radar + Detection + Segmentation System

 1. System Description

- Real-time person detection using YOLOv8 segmentation
- Distance estimation using bounding box width
- Radar-style threat detection overlay
- Threat zone classification (high, medium, low)
- Contour-based visual segmentation
- Unique ID assignment per person

2. Distance Estimation (Pinhole Camera Model)

To estimate distance from camera to person:

    Distance (in meters) = (Real Width × Focal Length) / (Width in Pixels × 100)

Where:
- Real Width = 50 cm (average shoulder width)
- Focal Length = 450 pixels (camera-specific)
- Width in Pixels = pixel width of the detected bounding box

3. Radar Angle Mapping

Maps the center x-coordinate of a detected person to an angle in the radar:

    Angle (in degrees) = (x_center / frame_width) × 180
4. Polar to Cartesian Conversion for Radar Plotting

Each detected person is plotted on a semi-circular radar overlay using:

    x = cx + r × cos(angle)
    y = cy - r × sin(angle)

Where:
- cx, cy = radar center
- r = radar radius scaled by distance
- angle = converted to radians

5. Threat Classification

Distance thresholds determine threat level:

    High:    0 to 1.5 meters
    Medium:  1.5 to 3.0 meters
    Low:     greater than 3.0 meters

Each threat level is color coded on the radar:
- Red = High
- Yellow = Medium
- Blue = Low

6. ID Assignment

Each new detection is matched to existing IDs using proximity:

    if abs(cx_new - cx_old) < 60 and abs(cy_new - cy_old) < 60:
        reuse ID
    else:
        assign new ID

7. Segmentation Contour Drawing

When masks are returned by YOLOv8:
- Binary masks are cleaned using morphological operations
- External contours are extracted using OpenCV's `findContours`
- Only large contours (> 2000 pixels) are drawn

Main.py – Simplified Version with Saved Images and Optimized Overlay

Includes:
- FPS tracking
- Clean tag overlays with ID and distance
- Saves cropped person images when a new ID is created

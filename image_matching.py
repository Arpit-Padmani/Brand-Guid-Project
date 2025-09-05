import cv2
import numpy as np
import os

def find_logo_in_post(logo_path: str, post_path: str, threshold: float = 0.8) -> tuple[bool, tuple | None]:
    """
    Checks if a logo is present in a social media post image by trying different scales of the logo.

    Args:
        logo_path (str): The file path to the logo image.
        post_path (str): The file path to the post image.
        threshold (float): The correlation threshold to determine a match.
                          A value of 1.0 is a perfect match.

    Returns:
        tuple[bool, tuple | None]: A tuple containing:
            - True if the logo is found, False otherwise.
            - A tuple with the (x, y) coordinates of the top-left corner of the best match, or None if no match is found.
    """
    # Check if the files exist
    if not os.path.exists(logo_path):
        print(f"Error: Logo file not found at '{logo_path}'")
        return False, None
    if not os.path.exists(post_path):
        print(f"Error: Post file not found at '{post_path}'")
        return False, None

    # Load the images
    logo_original = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    post = cv2.imread(post_path, cv2.IMREAD_UNCHANGED)

    if logo_original is None:
        print(f"Error: Failed to load logo image from '{logo_path}'. Check file format and integrity.")
        return False, None
    if post is None:
        print(f"Error: Failed to load post image from '{post_path}'. Check file format and integrity.")
        return False, None
    
    # Handle transparent backgrounds by creating a mask for the logo
    if logo_original.shape[2] == 4:
        logo_mask_original = logo_original[:, :, 3]
        logo_gray = cv2.cvtColor(logo_original, cv2.COLOR_BGR2GRAY)
    else:
        logo_mask_original = None
        logo_gray = cv2.cvtColor(logo_original, cv2.COLOR_BGR2GRAY)
        
    post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    # Initialize variables to track the best match
    best_match_val = -1
    best_match_loc = None
    best_scale = 1.0

    # Loop over different scales of the logo to find a match
    # A smaller step size (e.g., 0.05) would be more accurate but slower.
    for scale in np.linspace(1.0, 0.2, 20):
        # Calculate the new dimensions
        new_width = int(logo_gray.shape[1] * scale)
        new_height = int(logo_gray.shape[0] * scale)

        # Skip if the resized logo is too small or larger than the post
        if new_width == 0 or new_height == 0 or new_height > post_gray.shape[0] or new_width > post_gray.shape[1]:
            continue

        # Resize the logo and its mask based on the current scale
        resized_logo_gray = cv2.resize(logo_gray, (new_width, new_height))
        
        resized_mask = None
        if logo_mask_original is not None:
            resized_mask = cv2.resize(logo_mask_original, (new_width, new_height))
            
        # Perform template matching with the resized logo
        result = cv2.matchTemplate(post_gray, resized_logo_gray, cv2.TM_CCOEFF_NORMED, mask=resized_mask)
        
        # Find the max correlation value and its location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Update the best match if a higher correlation is found
        if max_val > best_match_val:
            best_match_val = max_val
            best_match_loc = max_loc
            best_scale = scale

    print(f"Highest correlation found across all scales: {best_match_val:.2f} at scale {best_scale:.2f}")

    # Check if the highest correlation is above the threshold
    if best_match_val >= threshold:
        print(f"Logo found! Match score: {best_match_val:.2f} (Threshold: {threshold:.2f})")
        # Get the dimensions of the best-matched logo
        best_logo_width = int(logo_original.shape[1] * best_scale)
        best_logo_height = int(logo_original.shape[0] * best_scale)
        
        # The top-left corner of the matched area
        top_left = best_match_loc
        bottom_right = (top_left[0] + best_logo_width, top_left[1] + best_logo_height)
        
        # Draw a rectangle to visualize the match
        cv2.rectangle(post, top_left, bottom_right, (0, 255, 0), 2)
        
        # Display the result (optional, for verification)
        cv2.imshow('Match Found', post)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True, top_left
    else:
        print(f"Logo not found. Match score: {best_match_val:.2f} (Threshold: {threshold:.2f})")
        return False, None

# Example usage of the function
if __name__ == "__main__":
    # IMPORTANT: Replace 'logo.png' and 'post.png' with your actual file paths
    logo_file = "only_logog.png"
    post_file = "check.png"

    is_present, location = find_logo_in_post(logo_file, post_file)

    if is_present:
        print(f"The logo was successfully detected at coordinates: {location}")
    else:
        print("The logo was not detected in the post.")

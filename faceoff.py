## FaceOff 3.1
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from typing import Literal, Optional
from pydantic import BaseModel, Field
import numpy as np
import mediapipe as mp
from PIL import Image
import cv2
from invokeai.app.models.image import ImageCategory, ImageField, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)


class PILInvocationConfig(BaseModel):
    """Helper class to provide all PIL invocations with additional config"""

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["PIL", "image"],
            },
        }


class FaceOffOutput(BaseInvocationOutput):
    """Base class for FaceOff Output"""

    # fmt: off
    type:              Literal["face_off_output"] = "face_off_output"
    bounded_image:     ImageField = Field(default=None, description="Original image bound, cropped, and resized")
    width:             int = Field(description="The width of the bounded image in pixels")
    height:            int = Field(description="The height of the bounded image in pixels")
    mask:              ImageField = Field(default=None, description="The output mask")
    x:                 int = Field(description="The x coordinate of the bounding box's left side")
    y:                 int = Field(description="The y coordinate of the bounding box's top side")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "bounded_image", "width", "height", "mask", "x", "y"]}


class FaceOffInvocation(BaseInvocation, PILInvocationConfig):
    """bound, extract, and mask a face from an image using MediaPipe detection"""

    # fmt: off
    type: Literal["face_off"] = "face_off"

    # Inputs
    image:               Optional[ImageField]  = Field(default=None, description="Image for face detection")
    face_id:             int = Field(default=0, description="0 for first detected face, single digit for one specific. Multiple faces not supported. Find a face's ID with FaceIdentifier node.")
    faces:               int = Field(default=4, description="Maximum number of faces to detect")
    minimum_confidence:  float = Field(default=0.5, description="Minimum confidence for face detection (lower if detection is failing)")
    x_offset:            float = Field(default=0.0, description="X-axis offset of the mask")
    y_offset:            float = Field(default=0.0, description="Y-axis offset of the mask")
    padding:             int = Field(default=0, description="All-axis padding around the mask in pixels")
    scale_factor:        int = Field(default=2, description="Factor to scale the bounding box by before outputting")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "FaceOff",
                "tags": ["image", "faceoff", "face", "mask"]
            },
        }

    def generate_face_box_mask(self, pil_image):
        # Convert the PIL image to a NumPy array.
        np_image = np.array(pil_image, dtype=np.uint8)

        # Check if the input image has four channels (RGBA).
        if np_image.shape[2] == 4:
            # Convert RGBA to RGB by removing the alpha channel.
            np_image = np_image[:, :, :3]

        # Create a FaceMesh object for face landmark detection and mesh generation.
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.faces,
            min_detection_confidence=self.minimum_confidence,
            min_tracking_confidence=self.minimum_confidence
        )

        # Detect the face landmarks and mesh in the input image.
        results = face_mesh.process(np_image)

        # Check if any face is detected.
        if results.multi_face_landmarks:
            face_id_counter = 1  # Start face ID counter from 1
            # If face_id is higher than faces to detect
            if self.face_id > self.faces:
                raise ValueError("Requested Face ID is higher than maximum faces to detect. Increase the Faces input value.")
            # If face_id index # is out of results.multi_face_landmarks scope
            elif not (0 <= self.face_id < (len(results.multi_face_landmarks) + 1)):
                raise ValueError("Requested Face ID is not in the list of detected faces.")
            # If face_id is set to 0, use the first detected face.
            elif self.face_id == 0:
                face_landmarks = results.multi_face_landmarks[0]  # Use the first detected face.
            else:
                # Search for the face_id in the detected faces.
                for face_landmarks in results.multi_face_landmarks:
                    if face_id_counter == self.face_id:
                        break
                    face_id_counter += 1

                # Check if the requested face_id is found.
                if face_id_counter != self.face_id:
                    raise ValueError("Requested face ID not found in the detected faces.")

            # Get the bounding box of the face mesh.
            x_coordinates = [landmark.x for landmark in face_landmarks.landmark]
            y_coordinates = [landmark.y for landmark in face_landmarks.landmark]
            x_min, x_max = min(x_coordinates), max(x_coordinates)
            y_min, y_max = min(y_coordinates), max(y_coordinates)

            # Calculate the width and height of the face mesh.
            mesh_width = int((x_max - x_min) * np_image.shape[1])
            mesh_height = int((y_max - y_min) * np_image.shape[0])

            # Get the center of the face.
            x_center = np.mean([landmark.x * np_image.shape[1] for landmark in face_landmarks.landmark])
            y_center = np.mean([landmark.y * np_image.shape[0] for landmark in face_landmarks.landmark])

            # Generate a binary face mask using the face mesh.
            mask_image = np.ones(np_image.shape[:2], dtype=np.uint8) * 255
            face_landmark_points = np.array(
                [[landmark.x * np_image.shape[1], landmark.y * np_image.shape[0]] for landmark in face_landmarks.landmark]
            )

            # Apply the scaling offsets to the face landmark points with a multiplier.
            scale_multiplier = 0.2
            x_center = np.mean(face_landmark_points[:, 0])
            y_center = np.mean(face_landmark_points[:, 1])
            x_scaled = face_landmark_points[:, 0] + scale_multiplier * self.x_offset * (face_landmark_points[:, 0] - x_center)
            y_scaled = face_landmark_points[:, 1] + scale_multiplier * self.y_offset * (face_landmark_points[:, 1] - y_center)

            convex_hull = cv2.convexHull(np.column_stack((x_scaled, y_scaled)).astype(np.int32))
            cv2.fillConvexPoly(mask_image, convex_hull, 0)

            # Convert the binary mask image to a PIL Image.
            mask_pil = Image.fromarray(mask_image, mode='L')

            return mask_pil, x_center, y_center, mesh_width, mesh_height

        else:
            raise ValueError("No face detected in the input image.")

    def invoke(self, context: InvocationContext) -> FaceOffOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        # Generate the face box mask and get the center of the face.
        mask_pil, center_x, center_y, mesh_width, mesh_height = self.generate_face_box_mask(image)

        # Determine the minimum size of the square crop
        min_size = min(mask_pil.width, mask_pil.height)

        # Calculate the crop boundaries for the output image and mask.
        mesh_width+=(128 + self.padding) # add pixels to account for mask variance
        mesh_height+=(128 + self.padding) # add pixels to account for mask variance
        crop_size = min(max(mesh_width, mesh_height, 128), min_size)  # Choose the smaller of the two (given value or face mask size)
        if crop_size > 128:
            crop_size = (crop_size + 7) // 8 * 8   # Ensure crop side is multiple of 8

        # Calculate the actual crop boundaries within the bounds of the original image.
        x_min = int(center_x - crop_size / 2)
        y_min = int(center_y - crop_size / 2)
        x_max = int(center_x + crop_size / 2)
        y_max = int(center_y + crop_size / 2)

        # Adjust the crop boundaries to stay within the original image's dimensions
        if x_min < 0:
            context.services.logger.warning(f'FaceOff --> -X-axis padding reached image edge.')
            x_max -= x_min
            x_min = 0
        elif x_max > mask_pil.width:
            context.services.logger.warning(f'FaceOff --> +X-axis padding reached image edge.')
            x_min -= (x_max - mask_pil.width)
            x_max = mask_pil.width

        if y_min < 0:
            context.services.logger.warning(f'FaceOff --> +Y-axis padding reached image edge.')
            y_max -= y_min
            y_min = 0
        elif y_max > mask_pil.height:
            context.services.logger.warning(f'FaceOff --> -Y-axis padding reached image edge.')
            y_min -= (y_max - mask_pil.height)
            y_max = mask_pil.height

        # Ensure the crop is square and adjust the boundaries if needed
        if x_max - x_min != crop_size:
            context.services.logger.warning(f'FaceOff --> Limiting x-axis padding to constrain bounding box to a square.')
            diff = crop_size - (x_max - x_min)
            x_min -= diff // 2
            x_max += diff - diff // 2

        if y_max - y_min != crop_size:
            context.services.logger.warning(f'FaceOff --> Limiting y-axis padding to constrain bounding box to a square.')
            diff = crop_size - (y_max - y_min)
            y_min -= diff // 2
            y_max += diff - diff // 2

        context.services.logger.info(f'FaceOff --> Calculated bounding box (8 multiple): {crop_size}')
        context.services.logger.info(f'FaceOff --> Scale factor: {self.scale_factor}')
        if self.scale_factor == 0:
            context.services.logger.info(f'FaceOff --> Scaled bounding box: {crop_size}')
        else:
            context.services.logger.info(f'FaceOff --> Scaled bounding box: {crop_size * self.scale_factor}')

        # Crop the output image to the specified size with the center of the face mesh as the center.
        mask_pil = mask_pil.crop((x_min, y_min, x_max, y_max))
        bounded_image = image.crop((x_min, y_min, x_max, y_max))

        # Resize images by a factor.
        if self.scale_factor > 0:
            new_size = (mask_pil.width * self.scale_factor, mask_pil.height * self.scale_factor)
            mask_pil = mask_pil.resize(new_size)
            bounded_image_np = np.array(bounded_image)
            bounded_image_np = cv2.resize(bounded_image_np, new_size, interpolation=cv2.INTER_LANCZOS4)
            bounded_image = Image.fromarray(bounded_image_np)

        # Convert the input image to RGBA mode to ensure it has an alpha channel.
        bounded_image = bounded_image.convert("RGBA")

        bounded_image_dto = context.services.images.create(
            image=bounded_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.OTHER,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )
        mask_dto = context.services.images.create(
            image=mask_pil,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return FaceOffOutput(
            bounded_image=ImageField(image_name=bounded_image_dto.image_name),
            width=bounded_image_dto.width,
            height=bounded_image_dto.height,
            mask=ImageField(image_name=mask_dto.image_name),
            x=x_min,
            y=y_min,
        )

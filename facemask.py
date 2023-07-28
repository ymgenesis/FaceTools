## FaceMask 3.1
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from typing import Literal, Optional
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
import cv2
import mediapipe as mp
import numpy as np
from invokeai.app.invocations.baseinvocation import (BaseInvocation,
                                                     BaseInvocationOutput,
                                                     InvocationConfig,
                                                     InvocationContext)
from invokeai.app.models.image import (ImageCategory, ImageField,
                                            ResourceOrigin)


class PILInvocationConfig(BaseModel):
    """Helper class to provide all PIL invocations with additional config"""

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["PIL", "image"],
            },
        }


class FaceMaskOutput(BaseInvocationOutput):
    """Base class for FaceMask output"""

    # fmt: off
    type:       Literal["face_mask_output"] = "face_mask_output"
    image:      ImageField = Field(default=None, description="The output image")
    width:      int = Field(description="The width of the image in pixels")
    height:     int = Field(description="The height of the image in pixels")
    mask:       ImageField = Field(default=None, description="The output mask")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "image", "width", "height", "mask"]}


class FaceMaskInvocation(BaseInvocation, PILInvocationConfig):
    """Face mask creation using mediapipe face detection"""

    # fmt: off
    type: Literal["face_mask_detection"] = "face_mask_detection"

    # Inputs
    image:          Optional[ImageField]  = Field(default=None, description="Image to face detect")
    faces:          int = Field(default=1, description="Maximum number of faces to detect")
    x_offset:       float = Field(default=0.0, description="Offset for the X-axis of the face mask")
    y_offset:       float = Field(default=0.0, description="Offset for the Y-axis of the face mask")
    invert_mask:    bool = Field(default=False, description="Toggle to invert the mask")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "FaceMask",
                "tags": ["image", "face", "mask"]
            },
        }

    def generate_face_masks(self, pil_image):
        # Convert the PIL image to a NumPy array.
        np_image = np.array(pil_image, dtype=np.uint8)

        # Check if the input image has four channels (RGBA).
        if np_image.shape[2] == 4:
            # Convert RGBA to RGB by removing the alpha channel.
            np_image = np_image[:, :, :3]

        # Create a FaceMesh object for face landmark detection and mesh generation.
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.faces, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Detect the face landmarks and mesh in the input image.
        results = face_mesh.process(np_image)

        # Generate a binary face mask using the face mesh.
        mask_image = np.zeros_like(np_image[:, :, 0])
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_landmark_points = np.array(
                    [[landmark.x * np_image.shape[1],
                      landmark.y * np_image.shape[0]]
                     for landmark in face_landmarks.landmark])

                # Apply the scaling offsets to the face landmark points.
                scale_multiplier = 0.2
                x_center = np.mean(face_landmark_points[:, 0])
                y_center = np.mean(face_landmark_points[:, 1])
                x_scaled = face_landmark_points[:, 0] + scale_multiplier * self.x_offset * (
                    face_landmark_points[:, 0] - x_center)
                y_scaled = face_landmark_points[:, 1] + scale_multiplier * self.y_offset * (
                    face_landmark_points[:, 1] - y_center)

                convex_hull = cv2.convexHull(np.column_stack(
                    (x_scaled, y_scaled)).astype(
                    np.int32))
                cv2.fillConvexPoly(mask_image, convex_hull, 255)

            # Convert the binary mask image to a PIL Image.
            mask_pil = Image.fromarray(mask_image, mode='L')

            return mask_pil

        else:
            raise ValueError("Failed to detect 1 or more faces in the image.")
            context.services.logger.warning('Failed to detect 1 or more faces in the image.')

    def invoke(self, context: InvocationContext) -> FaceMaskOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        # Generate the face mesh mask.
        mask_pil = self.generate_face_masks(image)
        if self.invert_mask:
            mask_pil = ImageOps.invert(mask_pil)

        # Create an RGBA image with transparency
        image = image.convert("RGBA")

        image_dto = context.services.images.create(
            image=image,
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

        return FaceMaskOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            mask=ImageField(image_name=mask_dto.image_name),
        )

## FaceIdentifier 1.0
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from typing import Literal, Optional, Dict
from PIL import Image, ImageOps, ImageDraw
from pydantic import BaseModel, Field
import cv2
import mediapipe as mp
import numpy as np
from invokeai.app.invocations.baseinvocation import (BaseInvocation,
                                                     BaseInvocationOutput,
                                                     InvocationConfig,
                                                     InvocationContext)
from invokeai.app.models.image import (ImageCategory, ImageField,
                                            ResourceOrigin, ImageOutput)


class PILInvocationConfig(BaseModel):
    """Helper class to provide all PIL invocations with additional config"""

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["PIL", "image"],
            },
        }


class FaceIdentifierInvocation(BaseInvocation, PILInvocationConfig):
    """Outputs an image with detected face IDs printed on each face. For use with other FaceTools."""

    # fmt: off
    type: Literal["face_identifier"] = "face_identifier"

    # Inputs
    image:                Optional[ImageField]  = Field(default=None, description="Image to face detect")
    faces:                int = Field(default=4, description="Maximum number of faces to detect")
    minimum_confidence:   float = Field(default=0.5, description="Minimum confidence for face detection (lower if detection is failing)")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "FaceIdentifier",
                "tags": ["image", "face", "identifier"]
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
            max_num_faces=self.faces, min_detection_confidence=self.minimum_confidence, min_tracking_confidence=self.minimum_confidence)

        # Detect the face landmarks and mesh in the input image.
        results = face_mesh.process(np_image)

        # Initialize variables to store face IDs and their coordinates
        face_IDs = {}
        current_face_id = 1

        # Generate a binary face mask using the face mesh.
        mask_image = np.zeros_like(np_image[:, :, 0])
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_landmark_points = np.array(
                    [[landmark.x * np_image.shape[1],
                      landmark.y * np_image.shape[0]]
                     for landmark in face_landmarks.landmark])

                x_center = np.mean(face_landmark_points[:, 0])
                y_center = np.mean(face_landmark_points[:, 1])

                # Add face ID and its coordinates to the dictionary
                face_IDs[current_face_id] = {"x_center": x_center, "y_center": y_center}
                # Increment the face ID for the next face
                current_face_id += 1

            # Convert the binary mask image to a PIL Image.
            mask_pil = Image.fromarray(mask_image, mode='L')

            return mask_pil, face_IDs

        else:
            raise ValueError("Failed to detect 1 or more faces in the image.")
            context.services.logger.warning('Failed to detect 1 or more faces in the image.')

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        image = image.copy()

        # Generate the face mesh mask.
        mask_pil, face_IDs = self.generate_face_masks(image)

        # Paste face IDs on the output image
        draw = ImageDraw.Draw(image)
        for face_id, coords in face_IDs.items():
            x_coord = coords["x_center"]
            y_coord = coords["y_center"]
            draw.text((x_coord, y_coord), str(face_id), fill=(255, 255, 255, 255))

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

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )

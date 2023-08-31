## FaceMask 3.7
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from PIL import Image, ImageOps
import cv2
import mediapipe as mp
import numpy as np
from invokeai.app.models.image import (ImageCategory, ResourceOrigin)
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InputField,
    OutputField,
    invocation,
    invocation_output)


@invocation_output("face_mask_output")
class FaceMaskOutput(BaseInvocationOutput):
    """Base class for FaceMask output"""

    image:      ImageField = OutputField(description="The output image")
    width:      int = OutputField(description="The width of the image in pixels")
    height:     int = OutputField(description="The height of the image in pixels")
    mask:       ImageField = OutputField(description="The output mask")


@invocation("face_mask_detection", title="FaceMask", tags=["image", "face", "mask"], category="image")
class FaceMaskInvocation(BaseInvocation):
    """Face mask creation using mediapipe face detection"""

    image:                ImageField  = InputField(default=None, description="Image to face detect")
    face_ids:             str = InputField(default="0", description="0 for all faces, single digit for one, comma-separated list for multiple specific (1, 2, 4). Find face IDs with FaceIdentifier node.")
    faces:                int = InputField(default=4, description="Maximum number of faces to detect")
    minimum_confidence:   float = InputField(default=0.5, description="Minimum confidence for face detection (lower if detection is failing)")
    x_offset:             float = InputField(default=0.0, description="Offset for the X-axis of the face mask")
    y_offset:             float = InputField(default=0.0, description="Offset for the Y-axis of the face mask")
    invert_mask:          bool = InputField(default=False, description="Toggle to invert the mask")

    def scale_and_convex(self, np_image, face_landmark_points):
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
        cv2.fillConvexPoly(np_image, convex_hull, 255)

    def generate_face_masks(self, pil_image):
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

        # Generate a binary face mask using the face mesh.
        mask_image = np.zeros_like(np_image[:, :, 0])
        if results.multi_face_landmarks:
            face_id_counter = 1  # Start face ID counter from 1

            if str(self.face_ids) == '0':
                # If '0' is entered, mask all faces
                for face_landmarks in results.multi_face_landmarks:
                    face_landmark_points = np.array(
                        [[landmark.x * np_image.shape[1],
                          landmark.y * np_image.shape[0]]
                         for landmark in face_landmarks.landmark])

                    self.scale_and_convex(mask_image, face_landmark_points)

                    face_id_counter += 1  # Increment the face ID counter for the next face
            else:
                # If specific face IDs are provided, mask only those faces
                for face_landmarks in results.multi_face_landmarks:
                    if str(face_id_counter) in str(self.face_ids).split(', '):
                        face_landmark_points = np.array(
                            [[landmark.x * np_image.shape[1],
                              landmark.y * np_image.shape[0]]
                             for landmark in face_landmarks.landmark])

                        self.scale_and_convex(mask_image, face_landmark_points)

                    face_id_counter += 1  # Increment the face ID counter for the next face

            # Convert the binary mask image to a PIL Image.
            mask_pil = Image.fromarray(mask_image, mode='L')

            return mask_pil

        else:
            raise ValueError("Failed to detect 1 or more faces in the image.")


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
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
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

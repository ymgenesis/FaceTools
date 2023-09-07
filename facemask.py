## FaceMask 3.7
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from PIL import Image, ImageOps
import cv2
import mediapipe as mp
import numpy as np
import math
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

def cleanup_faces_list(orig):
    newlist = []
    for i in orig:
        should_add = True
        i_x_center = i["x_center"]
        i_y_center = i["y_center"]
        for j in newlist:
            face_center_x = j["x_center"]
            face_center_y = j["y_center"]
            face_radius_w = j["mesh_width"] / 2
            face_radius_h = j["mesh_height"] / 2
            # Determine if the center of the candidate i is inside the ellipse of the added face
            # p < 1 -> Inside
            # p = 1 -> Exactly on the ellipse
            # p > 1 -> Outside
            p = ((math.pow((i_x_center - face_center_x), 2) / math.pow(face_radius_w, 2)) +
                 (math.pow((i_y_center - face_center_y), 2) / math.pow(face_radius_h, 2)))

            if p < 1: # Inside of the already-added face's radius
                should_add = False
                break

        if should_add is True:
            newlist.append(i)

    newlist = sorted(newlist, key=lambda x: x['y_center'])
    newlist = sorted(newlist, key=lambda x: x['x_center'])

    # add a face_id for reference
    face_id_counter = 1
    for face in newlist:
        face["face_id"] = face_id_counter
        face_id_counter += 1

    return newlist

@invocation_output("face_mask_output")
class FaceMaskOutput(BaseInvocationOutput):
    """Base class for FaceMask output"""

    image:      ImageField = OutputField(description="The output image")
    width:      int = OutputField(description="The width of the image in pixels")
    height:     int = OutputField(description="The height of the image in pixels")
    mask:       ImageField = OutputField(description="The output mask")


@invocation("face_mask_detection", title="FaceMask", tags=["image", "face", "mask"], category="image", version="1.0.0")
class FaceMaskInvocation(BaseInvocation):
    """Face mask creation using mediapipe face detection"""

    image:                ImageField  = InputField(description="Image to face detect")
    face_ids:             str = InputField(default="0", description="0 for all faces, single digit for one, comma-separated list for multiple specific (1, 2, 4). Find face IDs with FaceIdentifier node.")
    faces:                int = InputField(default=4, description="Maximum number of faces to detect")
    minimum_confidence:   float = InputField(default=0.5, description="Minimum confidence for face detection (lower if detection is failing)")
    x_offset:             float = InputField(default=0.0, description="Offset for the X-axis of the face mask")
    y_offset:             float = InputField(default=0.0, description="Offset for the Y-axis of the face mask")
    chunk:                bool = InputField(default=False, description="Whether to bypass full image face detection and default to image chunking. Chunking will occur if no faces are found in the full image.")
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

    def generate_face_masks(self, pil_image, chunk_x_offset=0, chunk_y_offset=0):
        
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

                # Convert the binary mask image to a PIL Image.
                init_mask_pil = Image.fromarray(mask_image, mode='L')
                w, h = init_mask_pil.size
                mask_pil = Image.new(mode='L', size=(w + chunk_x_offset, h + chunk_y_offset), color=255)
                mask_pil.paste(init_mask_pil, (chunk_x_offset, chunk_y_offset))

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
                    
                    # Convert the binary mask image to a PIL Image.
                    init_mask_pil = Image.fromarray(mask_image, mode='L')
                    w, h = init_mask_pil.size
                    mask_pil = Image.new(mode='L', size=(w + chunk_x_offset, h + chunk_y_offset), color=255)
                    mask_pil.paste(init_mask_pil, (chunk_x_offset, chunk_y_offset))

                    face_id_counter += 1  # Increment the face ID counter for the next face

        return mask_pil

    def facemask(self, context: InvocationContext) -> FaceMaskOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        # Generate the face mesh mask
        if self.chunk == False:
            context.services.logger.info(f'FaceMask --> Attempting full image face detection.')
            result = self.generate_face_masks(image)           

        if self.chunk == True or len(result) == 0:
            context.services.logger.info(f'FaceMask --> Chunking image (chunk toggled on, or no face found in full image).')
            width, height = image.size
            image_chunks = []
            x_offsets = []
            y_offsets = []
            result = []

            if width == height:
                # We cannot better handle a case where the image is square
                raise
            elif width > height:
                # Landscape - slice the image horizontally
                fx = 0.0
                steps = int(width * 2 / height)
                while fx <= (width - height):
                    x = int(fx)
                    image_chunks.append(image.crop((x, 0, x + height - 1, height - 1)))
                    x_offsets.append(x)
                    y_offsets.append(0)
                    fx += (width - height) / steps
                    context.services.logger.info(f'FaceMask --> Chunk starting at x = {x}')
            elif height > width:
                # Portrait - slice the image vertically
                fy = 0.0
                steps = int(height * 2 / width)
                while fy <= (height - width):
                    y = int(fy)
                    image_chunks.append(image.crop((0, y, width - 1, y + width - 1)))
                    x_offsets.append(0)
                    y_offsets.append(y)
                    fy += (height - width) / steps
                    context.services.logger.info(f'FaceMask --> Chunk starting at y = {y}')

            for idx in range(len(image_chunks)):
                context.services.logger.info(f'FaceMask --> Evaluating faces in chunk {idx}')
                mask_pil = self.generate_face_masks(image_chunks[idx], x_offsets[idx], y_offsets[idx])

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

    def invoke(self, context: InvocationContext) -> FaceMaskOutput:
        result = self.facemask(context)

        if result is None:
            image = context.services.images.get_pil_image(self.image.image_name)
            whitemask = Image.new("L", image.size, color=255)
            context.services.logger.info(f'FaceMask --> No face detected. Passing through original image.')

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
                image=whitemask,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.MASK,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
            )

            result = FaceMaskOutput(
                image=ImageField(image_name=image_dto.image_name),
                width=image_dto.width,
                height=image_dto.height,
                mask=ImageField(image_name=mask_dto.image_name),
            )
        
        return result
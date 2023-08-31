## FacePlace 1.8
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from PIL import Image
from invokeai.app.models.image import (ImageCategory, ResourceOrigin)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InvocationContext,
    InputField,
    invocation)


@invocation("face_place", title="FacePlace", tags=["image", "face", "place"], category="image")
class FacePlaceInvocation(BaseInvocation):
    """FacePlace node to place the a bounded face from FaceOff back onto the original image"""

    bounded_image:   ImageField = InputField(default=None, description="The bounded image to be placed on the original image")
    original_image:    ImageField = InputField(default=None, description="The original image to place the bounded image on")
    downscale_factor:  int = InputField(default=2, description="Factor to downscale the bounded image before placing")
    x:                 int = InputField(default=0, description="The x coordinate (top left corner) to place on the original image")
    y:                 int = InputField(default=0, description="The y coordinate (top left corner) to place on the original image")

    def create_alpha_mask(self, image):
        # Check the image mode to determine if it has an alpha channel.
        if image.mode == "RGBA":
            try:
                alpha = image.getchannel("A")  # Get the alpha channel (assuming RGBA image)
            except ValueError:
                raise ValueError("The image has no alpha channel (A).")
        else:
            # If the image is not in RGBA mode (e.g., RGB mode), create a mask of all opaque (white).
            alpha = Image.new("L", image.size, 255)

        return alpha

    def invoke(self, context: InvocationContext) -> ImageOutput:
        bounded_image = context.services.images.get_pil_image(self.bounded_image.image_name)
        original_image = context.services.images.get_pil_image(self.original_image.image_name)

        # Downscale the inpainted image by the given factor.
        if self.downscale_factor > 0:
            new_size = (int(bounded_image.width / self.downscale_factor), int(bounded_image.height / self.downscale_factor))
            bounded_image = bounded_image.resize(new_size)

        # Get the coordinates for placing the inpainted image on the original image.
        x_coord = self.x
        y_coord = self.y

        # Ensure the placement coordinates are within the bounds of the original image.
        x_max = x_coord + bounded_image.width
        y_max = y_coord + bounded_image.height
        if x_max > original_image.width:
            x_coord = original_image.width - bounded_image.width
        if y_max > original_image.height:
            y_coord = original_image.height - bounded_image.height

        # Create an alpha mask for the inpainted image.
        inpainted_alpha_mask = self.create_alpha_mask(bounded_image)

        # Create a copy of the original image.
        placed_image = original_image.copy()

        # Paste the inpainted image on the original image.
        placed_image.paste(bounded_image, (x_coord, y_coord), inpainted_alpha_mask)

        placed_image_dto = context.services.images.create(
            image=placed_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,            
        )

        return ImageOutput(
            image=ImageField(image_name=placed_image_dto.image_name),
            width=placed_image_dto.width,
            height=placed_image_dto.height,
        )

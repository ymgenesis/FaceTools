# FaceTools

A nodes extension for use with [InvokeAI](https://github.com/invoke-ai/InvokeAI "InvokeAI").

The usual inpainting technique of adding detail/changing faces in Canvas consists of resizing the bounding box around the head, drawing a mask on the face, and generating according to selected settings (denoise strength, steps, scheduler, etc.). As this sort of UI is not currently possible in the Experimental Node Editor, the FaceOff, FaceMask, and FacePlace nodes were created specifically to give you similar functionality through a semi-automated process.

## FaceOff

FaceOff mimics a user finding a face in an image and resizing the bounding box around the head in Canvas. Just as you would add more context inside the bounding box by making it larger in Canvas, the node gives you a padding input (in pixels) which will simultanesoly add more context, and increase the resolution of the bounding box so the face remains the same size inside it. The node also allows you to scale the bounding box by a factor to a higher resolution which may result in finer detail.

FaceOff will output the face in a bounded image, taking the face off of the original image for input into any node that accepts image inputs. The node also outputs a face mask with the dimensions of the bounded image. The X & Y outputs are for connecting to the X & Y inputs of FacePlace, which will place the bounded image back on the original image using these coordinates.

###### Inputs/Outputs

| Input | Description |
| -------- | ------------ |
| Image | Image for face detection |
| X Offset | X-axis offset of the mask |
| Y Offset | Y-axis offset of the mask |
| Padding | All-axis padding around the mask in pixels |
| Scale Factor | Factor to scale the bounding box by before outputting |

| Output | Description |
| -------- | ------------ |
| Bounded Image | Original image bound, cropped, and resized |
| Width | The width of the bounded image in pixels |
| Height | The height of the bounded image in pixels |
| Mask | The output mask |
| X | The x coordinate of the bounding box's left side |
| Y | The y coordinate of the bounding box's top side |



## FaceMask

FaceMask mimics a user drawing a mask on a face in an image in Canvas. If the detected mask is imperfect and strays too far outside/inside the face, the node gives you X and Y offsets to shrink/grow the mask by a multiplier. By default, the mask is created to protect the face, with only the surrounding area being affected by inpainting (ie. putting a face on another body). When the mask is inverted, the mask protects the surrounding area, with only the face being affected by inpainting (ie. putting a different face on the same body).

###### Inputs/Outputs

| Input | Description |
| -------- | ------------ |
| Image | Image for face detection |
| X Offset | X-axis offset of the mask |
| Y Offset | Y-axis offset of the mask |
| Invert Mask | Toggle to invert the face mask |

| Output | Description |
| -------- | ------------ |
| Image | The original image |
| Width | The width of the image in pixels |
| Height | The height of the image in pixels |
| Mask | The output face mask |



## FacePlace

FacePlace is a simple node that will take in the bounded image from FaceOff (either directly, or after processing with other nodes), the original image to place the bounded image back on to, the factor to downscale the bounded image by (if previously upscaled in FaceOff or other nodes), and the X & Y coordinate outputs from FaceOff.

###### Inputs/Outputs

| Input | Description |
| -------- | ------------ |
| Bounded Image | The bounded image to be placed on the original image |
| Original Image | The original image to place the bounded image on |
| Downscale Factor | Factor to downscale the bounded image before placing |
| X | The x coordinate (top left corner) to place on the original image |
| Y | The y coordinate (top left corner) to place on the original image |

| Output | Description |
| -------- | ------------ |
| Image | The full image with the face placed on |
| Width | The width of the image in pixels |
| Height | The height of the image in pixels |
<hr>

# Tips

- Non-inpainting models may struggle to paint/generate correctly around faces
- If you're getting lines or dots around the face, increase the seam/edge size. Higher numbers should take them away. Higher bounding box resolutions may increase the chance to get white lines or dots, and a higher seam/edge size may be necessary. These artefacts are the result of white areas of the mask making their way into the black areas.
- If the resulting face seems out of place pasted back on the original image (ie. too large, not proportional), add more padding on the FaceOff node to give inpainting more context. Context and good prompting are important to keeping things proportional.
- If you find the mask is too big/small and going too far outside/inside the area you want to affect, adjust the x & y offsets to shrink/grow the mask area
- Make sure to match the scaling factors between the two nodes (unless introducing other upscaling/downscaling prior to FacePlace, in which case some math is necessary). If upscaling the original image between FaceOff and FacePlace to avoid downscaling the bounded image before placement, the X & Y coordinate outputs from FaceOff have to be multiplied with Multiply nodes by the same upscale factor you upscaled the original image by in between.
- Use lower inpaint strength to resemble aspects of the original face or surroundings. Higher strengths will make something new.
- mediapipe isn't good at detecting faces with lots of face paint, hair covering the face, etc. Anything that obstructs the face will likely result in no faces being detected
- If choosing 0 upscaling on FaceOff and upscaling the bounded image with something harsher like RealESRGAN before passing into inpaint, the edges of the bounded image may be noticeable after being placed back on the original image with FacePlace.
<hr>

# Usage Examples

### FaceMask

FaceMask default usage with the inpaint node (July 27, 2023)

[vid]

FaceMask inverted mask usage with the inpaint node (July 27, 2023)

[vid]

<hr>
### FaceOff

FaceOff usage with the inpaint node, strength at 0.5 (July 27, 2023)

[vid]

FaceOff usage with the inpaint node, strength at 1.0 (July 27, 2023)

[vid]

FaceOff usage with the inpaint node and Real-ESRGAN upscaling, strength at 1.0 (July 27, 2023)

[vid]
#pragma once

#include "gl_utils/gl.h"
#include "gl_utils/gl_debug_output.h"
#include "gl_utils/gl_shader_util.h"
#include "gl_utils/gl_pixel_format.h"

typedef struct Compute_Texture {
    GLuint id;
    GL_Pixel_Format format;

    i32 width;
    i32 heigth;
} Compute_Texture;

EXPORT void render_sci_texture(Compute_Texture texture, f32 min, f32 max);
EXPORT void compute_texture_bind(Compute_Texture texture, GLenum access, isize slot);
EXPORT void compute_texture_deinit(Compute_Texture* texture);
EXPORT Compute_Texture compute_texture_make_with(isize width, isize heigth, Image_Pixel_Format format, isize channels, const void* data);
EXPORT Compute_Texture compute_texture_make(isize width, isize heigth, Image_Pixel_Format type, isize channels);

EXPORT void compute_texture_set_pixels(Compute_Texture* texture, Image_Builder image);
EXPORT void compute_texture_get_pixels(Image_Builder* into, Compute_Texture texture);
EXPORT void compute_texture_set_pixels_converted(Compute_Texture* texture, Image_Builder image);
EXPORT void compute_texture_get_pixels_converted(Image_Builder* into, Compute_Texture texture);

EXPORT void render_screen_quad()
{
    static GLuint quadVAO = 0;
    static GLuint quadVBO = 0;
	if (quadVAO == 0)
	{
		f32 quadVertices[] = {
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(f32), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(f32), (void*)(3 * sizeof(f32)));
	}

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}


EXPORT void render_sci_texture(Compute_Texture texture, f32 min, f32 max)
{
    static Render_Shader sci_shader = {0};
    if(sci_shader.shader == 0)
    {
        Allocator_Set prev = allocator_set_default(allocator_get_static());
        Error error = render_shader_init_from_disk(&sci_shader, STRING("shaders/sci_color.frag_vert"));
        TEST_MSG(error_is_ok(error), "Error %s while loading shaders!", error_code(error));
        allocator_set(prev);
    }
    
    //platform_thread_sleep(1);
    compute_texture_bind(texture, GL_READ_ONLY, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    render_shader_set_f32(&sci_shader, "_min", min);
    render_shader_set_f32(&sci_shader, "_max", max);
    render_shader_use(&sci_shader);
	render_screen_quad();
            
    if(0)
        render_shader_deinit(&sci_shader);

}

EXPORT Compute_Texture compute_texture_make_with(isize width, isize heigth, Image_Pixel_Format format, isize channels, const void* data)
{
    GL_Pixel_Format pixel_format = gl_pixel_format_from_pixel_format(format, channels);
    ASSERT(pixel_format.unrepresentable == false);
    
    Compute_Texture tex = {0};
	glGenTextures(1, &tex.id);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex.id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, pixel_format.internal_format, (GLuint) width, (GLuint) heigth, 0, pixel_format.format, pixel_format.type, data);

    tex.format = pixel_format;
    tex.width = (i32) width;
    tex.heigth = (i32) heigth;

	glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

EXPORT Compute_Texture compute_texture_make(isize width, isize heigth, Image_Pixel_Format type, isize channels)
{
    return compute_texture_make_with(width, heigth, type, channels, NULL);
}

EXPORT void compute_texture_bind(Compute_Texture texture, GLenum access, isize slot)
{
	glBindImageTexture((GLuint) slot, texture.id, 0, GL_FALSE, 0, access, texture.format.internal_format);
    glBindTextureUnit((i32) slot, texture.id);
}

EXPORT void compute_texture_deinit(Compute_Texture* texture)
{
    glDeleteTextures(1, &texture->id);
    memset(texture, 0, sizeof *texture);
}

EXPORT void compute_texture_set_pixels(Compute_Texture* texture, Image_Builder image)
{
    compute_texture_deinit(texture);
    *texture = compute_texture_make_with(image.width, image.height, (Image_Pixel_Format) image.pixel_format, image_builder_channel_count(image), image.pixels);
}

EXPORT void compute_texture_get_pixels(Image_Builder* into, Compute_Texture texture)
{
    image_builder_init(into, into->allocator, texture.format.channels, texture.format.equivalent);
    image_builder_resize(into, (i32) texture.width, (i32) texture.heigth);

    glGetTextureImage(texture.id, 0, texture.format.format, texture.format.type, (GLsizei) image_builder_all_pixels_size(*into), into->pixels);
}

EXPORT void compute_texture_set_pixels_converted(Compute_Texture* texture, Image_Builder image)
{
    if(texture->width != image.width || texture->heigth != image.height)
    {
        Image_Pixel_Format prev_pixe_format = texture->format.equivalent;
        isize prev_channel_count = texture->format.channels;

        compute_texture_deinit(texture);
        *texture = compute_texture_make(image.width, image.height, prev_pixe_format, prev_channel_count);
    }

    GL_Pixel_Format gl_format = gl_pixel_format_from_pixel_format((Image_Pixel_Format) image.pixel_format, image_builder_channel_count(image));
    glTextureSubImage2D(texture->id, 0, 0, 0, image.width, image.height, gl_format.format, gl_format.type, image.pixels);
}

EXPORT void compute_texture_get_pixels_converted(Image_Builder* into, Compute_Texture texture)
{
    image_builder_resize(into, (i32) texture.width, (i32) texture.heigth);
    GL_Pixel_Format gl_format = gl_pixel_format_from_pixel_format((Image_Pixel_Format) into->pixel_format, image_builder_channel_count(*into));
    
    glGetTextureImage(texture.id, 0, gl_format.format, gl_format.type, (GLsizei) image_builder_all_pixels_size(*into), into->pixels);
}
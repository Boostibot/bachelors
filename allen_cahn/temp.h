
void run_func_transfer(void* context)
{
    GLFWwindow* window = context;
    App_State* app = (App_State*) glfwGetWindowUserPointer(window); (void) app;

    Render_Shader screen_shader = {0}; 
    Render_Shader compute_shader = {0};

    Error error = {0};
    error = ERROR_OR(error) render_shader_init_from_disk(&screen_shader, STRING("shaders/sci_color.frag_vert"));
    error = ERROR_OR(error) compute_shader_init_from_disk(&compute_shader, STRING("shaders/heat_trasnfer.comp"), WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, WORK_GROUP_SIZE_Z);
    TEST_MSG(error_is_ok(error), "Error while loading shaders!");

    Compute_Texture next_state = compute_texture_make(TEXTURE_WIDTH, TEXTURE_HEIGHT, GL_FLOAT, 1);
    Compute_Texture prev_state = compute_texture_make(TEXTURE_WIDTH, TEXTURE_HEIGHT, GL_FLOAT, 1);

    i64 frame_counter = 0;
    f64 frame_time_sum = 0;
    
    f64 fps_display_last_time_sum = 0;
    f64 fps_display_last_time = 0;
    i64 fps_display_last_frames = 0;
    f64 render_last_time = 0;

    f64 simulation_time_sum = 0;

	while (!glfwWindowShouldClose(window))
    {
        f64 now = clock_s();
        if(now - render_last_time > 1.0/RENDER_FREQ)
        {
            render_last_time = now;

            compute_texture_bind(next_state, GL_WRITE_ONLY, 0);
            platform_thread_sleep(1);
		    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            render_shader_use(&screen_shader);
		    render_screen_quad();
            
            glfwSwapBuffers(window);
        }

        if(now - fps_display_last_time > 1.0/FPS_DISPLAY_FREQ)
        {
            f64 time_sum_delta = frame_time_sum - fps_display_last_time_sum;
            f64 counter_delta = (f64) (frame_counter - fps_display_last_frames);
            f64 avg_fps = 0;
            if(time_sum_delta != 0)
            {
                avg_fps = counter_delta / time_sum_delta;
                glfwSetWindowTitle(window, format_ephemeral("%5lf fps", avg_fps).data);
            }

            fps_display_last_time = now;
            fps_display_last_frames = frame_counter;
            fps_display_last_time_sum = frame_time_sum;
        }

        if(app->is_in_step_mode == false || app->remaining_steps > 0.5)
        {
            compute_texture_bind(next_state, GL_WRITE_ONLY, frame_counter % 2);
            compute_texture_bind(prev_state, GL_READ_ONLY, (frame_counter + 1) % 2);

            app->remaining_steps -= 1;
            
		    glMemoryBarrier(GL_ALL_BARRIER_BITS);

            f64 frame_start_time = clock_s();
            render_shader_set_f32(&compute_shader, "dt", (f32) app->dt);
            render_shader_set_f32(&compute_shader, "alpha", (f32) app->alpha);
            render_shader_set_i32(&compute_shader, "SIZE_X", TEXTURE_WIDTH);
            render_shader_set_i32(&compute_shader, "SIZE_Y", TEXTURE_HEIGHT);
            compute_shader_dispatch(&compute_shader, TEXTURE_WIDTH, TEXTURE_HEIGHT, 1);
            
		    // make sure writing to image has finished before read
		    glMemoryBarrier(GL_ALL_BARRIER_BITS);
            f64 end_start_time = clock_s();

            f64 delta = end_start_time - frame_start_time;
            frame_time_sum += delta;
            frame_counter += 1;
            simulation_time_sum += app->dt;
        }
        
		glfwPollEvents();
    }

    compute_texture_deinit(&next_state);
    compute_texture_deinit(&prev_state);
    render_shader_deinit(&screen_shader);
    render_shader_deinit(&compute_shader);
}

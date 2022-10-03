@rem Note that stb_image_write was renamed to a .c file
rem @del *.png
zig build-exe -lc -I. -DSTB_IMAGE_WRITE_IMPLEMENTATION stb_image_write.c raytracer.zig -Drelease-fast=true
rem raytracer.exe
rem start render_05.png

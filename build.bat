@rem Note that stb_image_write was renamed to a .c file
@del *.png
zig run -lc -I. -DSTB_IMAGE_WRITE_IMPLEMENTATION stb_image_write.c raytracer.zig -Drelease-fast=true 
start render_05.png
zig build-exe -lc -I. -DSTB_IMAGE_WRITE_IMPLEMENTATION stb_image_write.c raytracer.zig -Drelease-fast=true
raytracer.exe

@rem Note that stb_image_write was renamed to a .c file
@del render.png
zig run raytracer.zig -Drelease-fast=true -lc -I. -DSTB_IMAGE_WRITE_IMPLEMENTATION stb_image_write.c
start render.png

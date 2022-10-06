@rem Note that stb_image_write was renamed to a .c file
@del *.png
@del raytracer.exe
@rem Note that linking crashes in zig 0.9.1 without -fno-lto in release mode

zig build-exe -lc -I. -DSTB_IMAGE_WRITE_IMPLEMENTATION stb_image_write.c raytracer.zig -O ReleaseFast -fno-lto

raytracer.exe
start 05_book1_final_random.png

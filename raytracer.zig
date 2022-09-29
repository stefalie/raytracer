const std = @import("std");

const stb = @cImport({
    @cUndef("STB_IMAGE_WRITE_IMPLEMENTATION");
    @cInclude("stb_image_write.c");
});

var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
const gpa = general_purpose_allocator.allocator();

var pcg = std.rand.Pcg.init(0x853c49e6748fea9b);

fn deg2Rad(deg: f32) f32 {
    return std.math.degreesToRadians(f32, deg);
}
const infinity = std.math.inf(f32);
fn randFloat() f32 {
    return pcg.random().float(f32);
}
fn randFloatRange(min: f32, max: f32) f32 {
    return min + (max - min) * randFloat();
}
fn clamp(x: f32, min: f32, max: f32) f32 {
    if (x < min) {
        return min;
    }
    if (x > max) {
        return max;
    }
    return x;
}

// Abuse __m128 as 3D vector, allows using the usual math operators (mostly).
const Vec3 = @Vector(4, f32);
fn vec(x: f32, y: f32, z: f32) Vec3 {
    return Vec3{ x, y, z, 0.0 };
}
fn scalar(s: f32) Vec3 {
    return vec(s, s, s);
}
fn dot(lhs: Vec3, rhs: Vec3) f32 {
    const m = lhs * rhs;
    return m[0] + m[1] + m[2];
    // Note that @reduce(.Add, lhs * rhs) won't work since the 4th lane can contain garbage.
}
fn cross(lhs: Vec3, rhs: Vec3) Vec3 {
    return vec(
        lhs.y * rhs.z - rhs.y * lhs.z,
        lhs.z * rhs.x - rhs.z * lhs.x,
        lhs.x * rhs.y - rhs.x * lhs.y,
    );
}
fn lengthSquared(v: Vec3) f32 {
    return dot(v, v);
}
fn length(v: Vec3) f32 {
    return @sqrt(lengthSquared(v));
}
fn normalize(v: Vec3) Vec3 {
    return v / scalar(length(v));
}
pub fn printVec(v: Vec3) void {
    std.debug.print("({}, {}, {})", .{ v[0], v[1], v[2] });
}
const zero = scalar(0.0);
const one = scalar(1.0);

const Color = Vec3;
const color = vec;
const black = scalar(0.0);
const white = scalar(1.0);
const red = color(1.0, 0.0, 0.0);
const green = color(0.0, 1.0, 0.0);
const blue = color(0.0, 0.0, 1.0);

const RGB8 = struct {
    r: u8,
    g: u8,
    b: u8,

    pub fn init(c: Color, samples_per_pixel: usize) RGB8 {
        const scaled_c = c / scalar(@intToFloat(f32, samples_per_pixel));
        return .{
            .r = @floatToInt(u8, 256.0 * clamp(scaled_c[0], 0.0, 0.999)),
            .g = @floatToInt(u8, 256.0 * clamp(scaled_c[1], 0.0, 0.999)),
            .b = @floatToInt(u8, 256.0 * clamp(scaled_c[2], 0.0, 0.999)),
        };
    }
};

const Ray = struct {
    origin: Vec3,
    dir: Vec3,

    pub fn init(origin: Vec3, dir: Vec3) Ray {
        return .{
            .origin = origin,
            .dir = dir,
        };
    }
    pub fn at(self: Ray, t: f32) Vec3 {
        return self.origin + scalar(t) * self.dir;
    }
};

const HitRecord = struct {
    pos: Vec3,
    normal: Vec3,
    t: f32,
    is_front_face: bool,

    pub fn setFaceNormal(self: *HitRecord, r: Ray, outward_normal: Vec3) void {
        self.is_front_face = dot(r.dir, outward_normal) < 0;
        self.normal = if (self.is_front_face) outward_normal else -outward_normal;
    }
};

const Sphere = struct {
    center: Vec3,
    radius: f32,

    pub fn hit(self: Sphere, r: Ray, t_min: f32, t_max: f32) ?HitRecord {
        const oc = r.origin - self.center;
        const a = lengthSquared(r.dir);
        const half_b = dot(oc, r.dir);
        const c = lengthSquared(oc) - self.radius * self.radius;
        const discriminant = half_b * half_b - a * c;
        if (discriminant < 0.0) {
            return null;
        }

        const sqrtd = @sqrt(discriminant);
        const b_ge_zero = half_b >= 0.0;
        var root = if (b_ge_zero) (-half_b - sqrtd) / a else c / (-half_b + sqrtd);
        if (root < t_min or root > t_max) {
            root = if (b_ge_zero) c / (-half_b - sqrtd) else (-half_b + sqrtd) / a;
            if (root < t_min or root > t_max) {
                return null;
            }
        }

        var rec: HitRecord = undefined;
        rec.t = root;
        rec.pos = r.at(rec.t);
        const outward_normal = (rec.pos - self.center) / scalar(self.radius);
        rec.setFaceNormal(r, outward_normal);
        return rec;
    }
};

const World = struct {
    spheres: std.ArrayList(Sphere),

    pub fn init() World {
        return .{ .spheres = std.ArrayList(Sphere).init(gpa) };
    }
    pub fn deinit(self: World) void {
        self.spheres.deinit();
    }

    pub fn hit(self: World, r: Ray, t_min: f32, t_max: f32) ?HitRecord {
        var result_rec: ?HitRecord = null;

        for (self.spheres.items) |s| {
            const closest_so_far = if (result_rec) |rec| rec.t else t_max;
            if (s.hit(r, t_min, closest_so_far)) |new_rec| {
                result_rec = new_rec;
            }
        }

        return result_rec;
    }
};

const Camera = struct {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,

    pub fn init(aspect_ratio: f32) Camera {
        const viewport_height = 2.0;
        const viewport_width = aspect_ratio * viewport_height;
        const focal_length = 1.0;

        var res: Camera = undefined;
        res.origin = zero;
        res.horizontal = vec(viewport_width, 0.0, 0.0);
        res.vertical = vec(0.0, viewport_height, 0.0);
        res.lower_left_corner = res.origin - scalar(0.5) * res.horizontal - scalar(0.5) * res.vertical - vec(0.0, 0.0, focal_length);
        return res;
    }

    pub fn getRay(self: Camera, u: f32, v: f32) Ray {
        const pixel_pos = self.lower_left_corner + scalar(u) * self.horizontal + scalar(v) * self.vertical;
        return Ray.init(self.origin, pixel_pos - self.origin);
    }
};

fn rayColor(world: *World, r: Ray) Color {
    const intersection = world.hit(r, 0.0, infinity);
    if (intersection) |hit| {
        return scalar(0.5) * (hit.normal + one);
    } else {
        const t = scalar(0.5 * (normalize(r.dir)[1] + 1.0));
        return (one - t) * white + t * color(0.5, 0.7, 1.0);
    }
}

pub fn main() !void {
    const aspect_ratio = 16.0 / 9.0;
    const cam = Camera.init(aspect_ratio);

    const image_width = 400;
    const image_height = @floatToInt(usize, @as(f32, image_width) / aspect_ratio);
    var image: [image_height][image_width]RGB8 = undefined;
    const samples_per_pixel = 100;

    var world = World.init();
    defer world.deinit();
    try world.spheres.append(.{
        .center = vec(0.0, -100.5, -1.0),
        .radius = 100.0,
    });
    try world.spheres.append(.{
        .center = vec(0.0, 0.0, -1.0),
        .radius = 0.5,
    });
    try world.spheres.append(.{
        .center = vec(0.5, 0.0, -1.5),
        .radius = 0.5,
    });
    try world.spheres.append(.{
        .center = vec(-0.5, 0.0, -0.5),
        .radius = 0.5,
    });

    var y: usize = 0;
    while (y < image_height) : (y += 1) {
        const remaining_lines = image_height - y;
        if (remaining_lines % 10 == 0) {
            std.debug.print("Scanlines remaining: {}\n", .{remaining_lines});
        }

        var x: usize = 0;
        while (x < image_width) : (x += 1) {
            var pixel_color = zero;

            var i: usize = 0;
            while (i < samples_per_pixel) : (i += 1) {
                const u = (@intToFloat(f32, x) + randFloat()) / @as(f32, image_width - 1);
                const v = (@intToFloat(f32, y) + randFloat()) / @as(f32, image_height - 1);
                const ray = cam.getRay(u, v);
                pixel_color += rayColor(&world, ray);
            }

            image[y][x] = RGB8.init(pixel_color, samples_per_pixel);
        }
    }

    stb.stbi_flip_vertically_on_write(1);
    _ = stb.stbi_write_png(
        "render.png",
        image_width,
        image_height,
        3,
        @ptrCast(*const anyopaque, &image),
        image_width * 3,
    );
}

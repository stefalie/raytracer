const std = @import("std");

const stb = @cImport({
    @cUndef("STB_IMAGE_WRITE_IMPLEMENTATION");
    @cInclude("stb_image_write.c");
});

const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    pub fn init(x: f32, y: f32, z: f32) Vec3 {
        return .{
            .x = x,
            .y = y,
            .z = z,
        };
    }
    pub fn zero() Vec3 {
        return .{
            .x = 0.0,
            .y = 0.0,
            .z = 0.0,
        };
    }
    pub fn ones() Vec3 {
        return .{
            .x = 1.0,
            .y = 1.0,
            .z = 1.0,
        };
    }
    pub fn neg(self: Vec3) Vec3 {
        return .{
            .x = -self.x,
            .y = -self.y,
            .z = -self.z,
        };
    }
    pub fn add(self: Vec3, rhs: Vec3) Vec3 {
        return .{
            .x = self.x + rhs.x,
            .y = self.y + rhs.y,
            .z = self.z + rhs.z,
        };
    }
    pub fn sub(self: Vec3, rhs: Vec3) Vec3 {
        return self.add(rhs.neg());
    }
    pub fn mul(self: Vec3, t: f32) Vec3 {
        return .{
            .x = self.x * t,
            .y = self.y * t,
            .z = self.z * t,
        };
    }
    pub fn div(self: Vec3, t: f32) Vec3 {
        return self.mul(1.0 / t);
    }
    //pub fn mulCWise(self: Vec3, rhs: Vec3) Vec3 {
    //    return .{
    //        .x = self.x * rhs.x,
    //        .y = self.y * rhs.y,
    //        .z = self.z * rhs.z,
    //    };
    //}
    //pub fn divCWise(self: Vec3, rhs: Vec3) Vec3 {
    //    return .{
    //        .x = self.x / rhs.x,
    //        .y = self.y / rhs.y,
    //        .z = self.z / rhs.z,
    //    };
    //}
    pub fn dot(self: Vec3, rhs: Vec3) f32 {
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z;
    }
    pub fn cross(self: Vec3, rhs: Vec3) Vec3 {
        return .{
            self.y * rhs.z - rhs.y * self.z,
            self.z * rhs.x - rhs.z * self.x,
            self.x * rhs.y - rhs.x * self.y,
        };
    }
    pub fn lengthSquared(self: Vec3) f32 {
        return self.dot(self);
    }
    pub fn length(self: Vec3) f32 {
        return @sqrt(self.lengthSquared());
    }
    pub fn normalized(self: Vec3) Vec3 {
        return self.div(self.length());
    }
    pub fn print(self: Vec3) void {
        std.debug.print("({}, {}, {})", .{ self.x, self.y, self.z });
    }
};

const Color = Vec3;

const RGB8 = struct {
    r: u8,
    g: u8,
    b: u8,

    pub fn init(c: Color) RGB8 {
        return .{
            .r = @floatToInt(u8, 255.999 * c.x),
            .g = @floatToInt(u8, 255.999 * c.y),
            .b = @floatToInt(u8, 255.999 * c.z),
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
        return self.origin.add(self.dir.mul(t));
    }
};

fn rayColor(world: *World, r: Ray) Color {
    //const sphere = Sphere{
    //    .center = Vec3.init(0.0, 0.0, -1.0),
    //    .radius = 0.5,
    //};
    //const intersection = sphere.hit(r, 0.0, 10.0);
    const intersection = world.hit(r, 0.0, 10.0);
    if (intersection) |hit| {
        return hit.normal.add(Vec3.ones()).mul(0.5);
    } else {
        const t = 0.5 * (r.dir.normalized().y + 1.0);
        return Color.ones().mul(1.0 - t).add(Color.init(0.5, 0.7, 1.0).mul(t));
    }
}

const HitRecord = struct {
    pos: Vec3,
    normal: Vec3,
    t: f32,
    is_front_face: bool,

    pub fn setFaceNormal(self: *HitRecord, r: Ray, outward_normal: Vec3) void {
        self.is_front_face = Vec3.dot(r.dir, outward_normal) < 0;
        self.normal = if (self.is_front_face) outward_normal else outward_normal.neg();
    }
};

const Sphere = struct {
    center: Vec3,
    radius: f32,

    pub fn hit(self: Sphere, r: Ray, t_min: f32, t_max: f32) ?HitRecord {
        const oc = r.origin.sub(self.center);
        const a = r.dir.lengthSquared();
        const half_b = Vec3.dot(oc, r.dir);
        const c = oc.lengthSquared() - self.radius * self.radius;
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
        const outward_normal = rec.pos.sub(self.center).div(self.radius);
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
        //var closest_so_far = t_max;

        for (self.spheres.items) |s| {
            const closest_so_far = if (result_rec) |rec| rec.t else t_max;
            if (s.hit(r, t_min, closest_so_far)) |new_rec| {
                result_rec = new_rec;
            }
        }

        return result_rec;
    }
};

pub fn main() !void {
    const aspect = 16.0 / 9.0;
    const image_width = 1600;
    const image_height = @floatToInt(usize, @as(f32, image_width) / aspect);
    var image: [image_height][image_width]RGB8 = undefined;

    const viewport_height = 2.0;
    const viewport_width = aspect * viewport_height;
    const focal_length = 1.0;

    const origin = Vec3.zero();
    const horizontal = Vec3.init(viewport_width, 0.0, 0.0);
    const vertical = Vec3.init(0.0, viewport_height, 0.0);
    const lower_left_corner = origin.sub(horizontal.mul(0.5)).sub(vertical.mul(0.5)).sub(Vec3.init(0.0, 0.0, focal_length));

    var world = World.init();
    defer world.deinit();
    try world.spheres.append(.{
        .center = Vec3.init(0.0, 0.0, -1.0),
        .radius = 0.5,
    });
    try world.spheres.append(.{
        .center = Vec3.init(0.5, 0.0, -1.5),
        .radius = 0.5,
    });
    try world.spheres.append(.{
        .center = Vec3.init(-0.5, 0.0, -0.5),
        .radius = 0.5,
    });

    var y: usize = 0;
    while (y < image_height) : (y += 1) {
        const remaining_lines = image_height - y;
        if (remaining_lines % 100 == 0) {
            std.debug.print("Scanlines remaining: {}\n", .{remaining_lines});
        }

        var x: usize = 0;
        while (x < image_width) : (x += 1) {
            const u = @intToFloat(f32, x) / @as(f32, image_width - 1);
            const v = @intToFloat(f32, y) / @as(f32, image_height - 1);

            const pixel_pos = lower_left_corner.add(horizontal.mul(u)).add(vertical.mul(v));
            const ray = Ray.init(origin, pixel_pos.sub(origin));
            const pixel_color = rayColor(&world, ray);

            image[y][x] = RGB8.init(pixel_color);
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

var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
const gpa = general_purpose_allocator.allocator();

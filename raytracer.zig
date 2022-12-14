const std = @import("std");

const stb = @cImport({
    @cUndef("STB_IMAGE_WRITE_IMPLEMENTATION");
    @cInclude("stb_image_write.c");
});

var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
const gpa = general_purpose_allocator.allocator();

const pi = std.math.pi;
fn deg2Rad(deg: f32) f32 {
    return deg / 180.0 * pi;
    // In Zig 10.0
    // return std.math.degreesToRadians(f32, deg);
}
const infinity = std.math.inf(f32);

// Abuse __m128 as 3D vector, allows using the usual math operators (mostly).
const Vec3 = @Vector(3, f32);
fn scalar(s: f32) Vec3 {
    return .{ s, s, s };
}
fn dot(lhs: Vec3, rhs: Vec3) f32 {
    // Note that this wouldn't work if we used @Vector(4, ...) because the 4th lane can contain garbage.
    return @reduce(.Add, lhs * rhs);
}
fn cross(lhs: Vec3, rhs: Vec3) Vec3 {
    return .{
        lhs[1] * rhs[2] - rhs[1] * lhs[2],
        lhs[2] * rhs[0] - rhs[2] * lhs[0],
        lhs[0] * rhs[1] - rhs[0] * lhs[1],
    };
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
fn nearZero(v: Vec3) bool {
    const threshold = 1.0e-8;
    return @reduce(.And, @fabs(v) < scalar(threshold));
}
fn reflect(v: Vec3, n: Vec3) Vec3 {
    return v - scalar(2.0 * dot(v, n)) * n;
}
fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) Vec3 {
    const cos_theta = std.math.min(dot(-uv, n), 1.0);
    const r_out_perp = scalar(etai_over_etat) * (uv + scalar(cos_theta) * n);
    const r_out_parallel = scalar(-@sqrt(@fabs(1.0 - lengthSquared(r_out_perp)))) * n;
    return r_out_perp + r_out_parallel;
}

const zero = scalar(0.0);
const one = scalar(1.0);
const unit_x = Vec3{ 1.0, 0.0, 0.0 };
const unit_y = Vec3{ 0.0, 1.0, 0.0 };
const unit_z = Vec3{ 0.0, 0.0, 1.0 };

const Color = Vec3;
const black = scalar(0.0);
const white = scalar(1.0);
const red = unit_x;
const green = unit_y;
const blue = unit_z;

const Rand = struct {
    pcg: std.rand.Pcg,

    fn init(seed: u64) Rand {
        return .{ .pcg = std.rand.Pcg.init(seed) };
    }

    fn uniformInt64(self: *Rand) u64 {
        return self.pcg.random().int(u64);
    }
    fn uniformFloat(self: *Rand) f32 {
        return self.pcg.random().float(f32);
    }
    fn uniformFloatInRange(self: *Rand, min: f32, max: f32) f32 {
        return min + (max - min) * self.uniformFloat();
    }

    fn uniformVector(self: *Rand) Vec3 {
        return .{
            self.uniformFloat(),
            self.uniformFloat(),
            self.uniformFloat(),
        };
    }
    fn uniformVectorInRange(self: *Rand, min: f32, max: f32) Vec3 {
        return .{
            self.uniformFloatInRange(min, max),
            self.uniformFloatInRange(min, max),
            self.uniformFloatInRange(min, max),
        };
    }

    // Archimedes' Hat-Box Theorem
    fn uniformUnitVector(self: *Rand) Vec3 {
        const w_z = 2.0 * self.uniformFloat() - 1.0;
        const r = @sqrt(1.0 - w_z * w_z);
        const theta = 2.0 * pi * self.uniformFloat();
        const w_x = r * @cos(theta);
        const w_y = r * @sin(theta);
        return .{ w_x, w_y, w_z };
    }

    fn uniformDisk(self: *Rand) Vec3 {
        const theta = 2.0 * pi * self.uniformFloat();
        const r = @sqrt(self.uniformFloat());
        return .{ r * @cos(theta), r * @sin(theta), 0.0 };
    }
    fn uniformSphere(self: *Rand) Vec3 {
        // Is it correct to use the cubic root of a uniform random number as radius?
        // Probably: https://stackoverflow.com/a/5408843
        const rand_unit = self.uniformUnitVector();
        const radius = std.math.pow(f32, self.uniformFloat(), 1.0 / 3.0);
        return scalar(radius) * rand_unit;
    }
    fn uniformSphereRejectionSampling(self: *Rand) Vec3 {
        return while (true) {
            const p = Vec3{
                self.uniformFloatInRange(-1.0, 1.0),
                self.uniformFloatInRange(-1.0, 1.0),
                self.uniformFloatInRange(-1.0, 1.0),
            };
            if (lengthSquared(p) < 1.0) {
                break p;
            }
        } else unreachable;
    }

    // normal needs to be normalized
    fn cosineWeightedHemisphere(self: *Rand, normal: Vec3) Vec3 {
        std.debug.assert(@fabs(length(normal) - 1.0) < 1.0e-6);
        // This is cosine weighted importance sampling of the direction.
        // (See: https://twitter.com/mmalex/status/1550765798263758848)
        return normal + self.uniformUnitVector();
    }
};
var global_rand = Rand.init(0x853c49e6748fea9b);

const RGB8 = struct {
    r: u8,
    g: u8,
    b: u8,

    pub fn init(color_acc: Color, num_samples: usize) RGB8 {
        const color_avg = color_acc / scalar(@intToFloat(f32, num_samples));
        const color_gamma_encdoed = @sqrt(color_avg); // poor man's gamma encoding
        return .{
            .r = @floatToInt(u8, 256.0 * std.math.clamp(color_gamma_encdoed[0], 0.0, 0.999)),
            .g = @floatToInt(u8, 256.0 * std.math.clamp(color_gamma_encdoed[1], 0.0, 0.999)),
            .b = @floatToInt(u8, 256.0 * std.math.clamp(color_gamma_encdoed[2], 0.0, 0.999)),
        };
    }
};

const Ray = struct {
    origin: Vec3,
    dir: Vec3,

    pub fn at(self: Ray, t: f32) Vec3 {
        return self.origin + scalar(t) * self.dir;
    }
};

const HitRecord = struct {
    pos: Vec3,
    normal: Vec3,
    t: f32,
    material: MaterialHandle,
    is_front_face: bool,

    pub fn setFaceNormal(self: *HitRecord, r: Ray, outward_normal: Vec3) void {
        self.is_front_face = dot(r.dir, outward_normal) < 0;
        self.normal = if (self.is_front_face) outward_normal else -outward_normal;
    }
};

const Sphere = struct {
    center: Vec3,
    radius: f32,
    material: MaterialHandle,

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

        // Stable quadratic equation solving.
        // See:
        // https://math.stackexchange.com/questions/866331/numerically-stable-algorithm-for-solving-the-quadratic-equation-when-a-is-very
        //
        // A few notes on the particular case of sphere <-> ray intersection:
        // - we only want positive roots
        // - a is always > 0
        // - if b < 0 (the sphere center is behind the origin)
        //   there is at most one positive root
        //   if it exists it's the larger root
        // - c can be zero when the origin of the ray is on the sphere
        //   (that can happen when two spheres touch)
        // - it can happen that both c and b == 0
        //   I think that is the case when two spheres overlap
        //   and the ray is tangential to the sphere we intersect against
        // - with the classical solution for the roots the version
        //   with - is always smaller than the version with +

        const b_ge_zero = half_b >= 0.0;
        const b_c_eq_zero = c == 0.0 and half_b == 0.0; // Can lead to NaN

        var root = if (b_ge_zero) (-half_b - sqrtd) / a else c / (-half_b + sqrtd);
        if (root < t_min or root >= t_max) {
            root = if (b_ge_zero and !b_c_eq_zero) c / (-half_b - sqrtd) else (-half_b + sqrtd) / a;
            if (root < t_min or root >= t_max) {
                return null;
            }
        }

        // This is way simpler, but it's less precise.
        //var root = (-half_b - sqrtd) / a;
        //if (root < t_min or root >= t_max) {
        //    root = (-half_b + sqrtd) / a;
        //    if (root < t_min or root >= t_max) {
        //        return null;
        //    }
        //}

        // TODO: Should I the less precise expression but use f64?
        // It can't cause NaN.

        var rec: HitRecord = undefined;
        rec.t = root;
        rec.pos = r.at(rec.t);
        rec.material = self.material;
        // Normalizing by dividing by radius is less precise than re-normalizing.
        //const outward_normal = (rec.pos - self.center) / scalar(self.radius);
        const outward_normal = normalize(rec.pos - self.center) * scalar(if (self.radius >= 0.0) 1.0 else -1.0);
        std.debug.assert(@fabs(length(outward_normal) - 1.0) < 1.0e-6);
        rec.setFaceNormal(r, outward_normal);
        return rec;
    }
};

const World = struct {
    camera: Camera,
    spheres: std.ArrayList(Sphere),
    materials: std.ArrayList(Material),

    pub fn init(cam: Camera) World {
        return .{
            .camera = cam,
            .spheres = std.ArrayList(Sphere).init(gpa),
            .materials = std.ArrayList(Material).init(gpa),
        };
    }
    pub fn deinit(self: *World) void {
        self.spheres.deinit();
        self.materials.deinit();
    }

    pub fn addSphere(self: *World, sphere: Sphere) void {
        self.spheres.append(sphere) catch unreachable;
    }
    pub fn addMaterial(self: *World, mat: Material) MaterialHandle {
        self.materials.append(mat) catch unreachable;
        return @intCast(MaterialHandle, self.materials.items.len - 1);
    }

    pub fn hit(self: *const World, r: Ray, t_min: f32, t_max: f32) ?HitRecord {
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

const Framebuffer = struct {
    width: usize,
    height: usize,
    image: std.ArrayList(RGB8),

    pub fn init(height: usize, aspect_ratio: f32) Framebuffer {
        const width = @floatToInt(usize, @intToFloat(f32, height) * aspect_ratio);
        var img = std.ArrayList(RGB8).init(gpa);
        img.resize(width * height) catch unreachable;
        return .{
            .width = width,
            .height = height,
            .image = img,
        };
    }
    pub fn deinit(self: Framebuffer) void {
        self.image.deinit();
    }

    pub fn setPixel(self: Framebuffer, x: usize, y: usize, value: RGB8) void {
        self.image.items[y * self.width + x] = value;
    }
};

const Camera = struct {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    aspect_ratio: f32,
    u: Vec3,
    v: Vec3,
    lens_radius: f32,

    pub fn init(
        look_from: Vec3,
        look_at: Vec3,
        up: Vec3,
        fovy: f32,
        aspect_ratio: f32,
        aperture: f32,
        focus_dist: f32,
    ) Camera {
        const theta = deg2Rad(fovy);
        const h = std.math.tan(theta * 0.5);
        const viewport_height = 2.0 * h;
        const viewport_width = aspect_ratio * viewport_height;

        const w = normalize(look_from - look_at);
        const u = normalize(cross(up, w));
        const v = cross(w, u);

        const horizontal = u * scalar(focus_dist * viewport_width);
        const vertical = v * scalar(focus_dist * viewport_height);
        const lower_left_corner = look_from - scalar(0.5) * horizontal - scalar(0.5) * vertical - scalar(focus_dist) * w;

        return .{
            .origin = look_from,
            .lower_left_corner = lower_left_corner,
            .horizontal = horizontal,
            .vertical = vertical,
            .aspect_ratio = aspect_ratio,
            .u = u,
            .v = v,
            .lens_radius = aperture * 0.5,
        };
    }

    pub fn getRay(self: Camera, s: f32, t: f32, rand: *Rand) Ray {
        const rd = scalar(self.lens_radius) * rand.uniformDisk();
        const offset = self.u * scalar(rd[0]) + self.v * scalar(rd[1]);
        const pixel_pos = self.lower_left_corner + scalar(s) * self.horizontal + scalar(t) * self.vertical;
        return Ray{
            .origin = self.origin + offset,
            .dir = pixel_pos - self.origin - offset,
        };
    }
};

const Scatter = struct {
    ray_out: Ray,
    attenuation: Color,
};

const Lambertian = struct {
    albedo: Vec3,

    pub fn scatter(self: Lambertian, hit: HitRecord, rand: *Rand) Scatter {
        // The pdf is cos(theta) / pi
        // cos cancels the cos in the lighting equation
        // A proper Lambertian BRDF is albeda / pi
        // The pis in the pdf and the BRDF cancel each other.
        var scatter_dir = rand.cosineWeightedHemisphere(hit.normal);
        if (nearZero(scatter_dir)) {
            scatter_dir = hit.normal;
        }
        return .{
            .ray_out = Ray{
                .origin = hit.pos,
                .dir = scatter_dir,
            },
            .attenuation = self.albedo, // missing 1 / pi
        };
    }
};
const Metal = struct {
    albedo: Vec3,
    fuzz: f32,

    pub fn scatter(self: Metal, r_in: Ray, hit: HitRecord, rand: *Rand) ?Scatter {
        // NOTE: reflected needs to be normalized because 'fuzz' is assumed to be added to a unit vector.
        const reflected = reflect(normalize(r_in.dir), hit.normal);
        const dir_out = reflected + scalar(self.fuzz) * rand.uniformSphere();

        // This will absorb the ray if the fuzziness reflects below the surface
        if (dot(dir_out, hit.normal) > 0.0) {
            return Scatter{
                .ray_out = Ray{
                    .origin = hit.pos,
                    .dir = dir_out,
                },
                .attenuation = self.albedo, // Does this BRDF integrate to 1 over the hemisphere?
            };
        } else {
            return null;
        }
    }
};
const Dielectric = struct {
    refraction_index: f32,

    pub fn scatter(self: Dielectric, r_in: Ray, hit: HitRecord, rand: *Rand) Scatter {
        const refraction_ratio = if (hit.is_front_face) (1.0 / self.refraction_index) else self.refraction_index;
        const unit_dir = normalize(r_in.dir);

        // TODO: cos_theta is also computed inside refract(...), only do it once.
        const cos_theta = std.math.min(dot(-unit_dir, hit.normal), 1.0);
        const sin_theta = @sqrt(1.0 - cos_theta * cos_theta);
        const cannot_refract = refraction_ratio * sin_theta > 1.0;

        // Read "Reflections and Refractions in Ray Tracing" about TIR etc.
        var out_dir: Vec3 = undefined;
        if (cannot_refract or reflectanceSchlickApprox(cos_theta, self.refraction_index) > rand.uniformFloat()) {
            out_dir = reflect(unit_dir, hit.normal);
        } else {
            out_dir = refract(unit_dir, hit.normal, refraction_ratio);
        }
        return .{
            .ray_out = Ray{
                .origin = hit.pos,
                .dir = out_dir,
            },
            .attenuation = white,
        };
    }

    // NOTE: This is sneaky. There is no comparison between eta_1 and eta_2.
    // This will only work if the normal always points toward the side where the refraction index is smaller.
    // Luckily this also ok for spheres with a negative radius (but only because of the way
    // the sphere <-> ray intersection routine computes the normal).
    fn reflectanceSchlickApprox(cosine: f32, ref_idx: f32) f32 {
        var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0 - r0) * std.math.pow(f32, 1.0 - cosine, 5.0);
    }
};
const Material = union(enum) {
    lambertian: Lambertian,
    metal: Metal,
    dielectric: Dielectric,

    pub fn scatter(self: Material, r_in: Ray, hit: HitRecord, rand: *Rand) ?Scatter {
        return switch (self) {
            .lambertian => |l| l.scatter(hit, rand),
            .metal => |m| m.scatter(r_in, hit, rand),
            .dielectric => |d| d.scatter(r_in, hit, rand),
        };
    }
};
const MaterialHandle = u16;

fn rayColor(world: *const World, r: Ray, depth: isize, rand: *Rand) Color {
    if (depth <= 0) {
        return black;
    }

    const intersection = world.hit(r, 0.001, infinity);
    if (intersection) |hit| {
        const mat = world.materials.items[hit.material];
        if (mat.scatter(r, hit, rand)) |scatter| {
            return scatter.attenuation * rayColor(world, scatter.ray_out, depth - 1, rand);
        } else {
            return black;
        }
    } else {
        const t = scalar(0.5 * (normalize(r.dir)[1] + 1.0));
        return (one - t) * white + t * Color{ 0.5, 0.7, 1.0 };
    }
}

const RenderCon = struct {
    framebuffer: Framebuffer,
    world: *const World,
};
const RenderChunk = struct {
    rand: Rand, // Makes it deterministic as long as the chunk size stays the same.
    y_start: usize,
    y_end: usize,
};

const RenderChunkQueue = struct {
    next_chunk_idx: std.atomic.Atomic(u32),
    chunks: std.ArrayList(RenderChunk),

    pub fn init() RenderChunkQueue {
        return .{
            .next_chunk_idx = std.atomic.Atomic(u32).init(0),
            .chunks = std.ArrayList(RenderChunk).init(gpa),
        };
    }
    pub fn deinit(self: RenderChunkQueue) void {
        self.chunks.deinit();
    }
};

fn renderThread(
    framebuffer: Framebuffer,
    world: *const World,
    queue: *RenderChunkQueue,
) void {
    const img_width = framebuffer.width;
    const img_height = framebuffer.height;

    while (true) {
        const chunk_idx = queue.next_chunk_idx.fetchAdd(1, .AcqRel);
        if (chunk_idx >= queue.chunks.items.len) {
            break;
        }

        std.debug.print("Thread {d: >5}:\tStarting chunk {d: >3}\n", .{ std.Thread.getCurrentId(), chunk_idx });
        var chunk = queue.chunks.items[chunk_idx];

        var y: usize = chunk.y_start;
        while (y < chunk.y_end) : (y += 1) {
            var x: usize = 0;
            while (x < img_width) : (x += 1) {
                var pixel_color = zero;

                var i: usize = 0;
                while (i < cfg.samples_per_pixel) : (i += 1) {
                    const u = (@intToFloat(f32, x) + chunk.rand.uniformFloat()) / @intToFloat(f32, img_width); // I think the book wrongly does -1 in the dividend.
                    const v = (@intToFloat(f32, y) + chunk.rand.uniformFloat()) / @intToFloat(f32, img_height);

                    const ray = world.camera.getRay(u, v, &chunk.rand);
                    pixel_color += rayColor(world, ray, cfg.max_depth, &chunk.rand);
                }

                framebuffer.setPixel(x, y, RGB8.init(pixel_color, cfg.samples_per_pixel));
            }
        }
    }

    std.debug.print("Thread {}:\tDone\n", .{std.Thread.getCurrentId()});
}

const Quality = enum {
    quicky,
    preview,
    fhd,
};

const quality = Quality.fhd;
const num_threads = 16;
const scanlines_per_chunk = 8;

const Config = struct {
    image_height: usize,
    samples_per_pixel: usize,
    max_depth: usize,
};
const cfg: Config = switch (quality) {
    .quicky => .{
        .image_height = 225,
        .samples_per_pixel = 1,
        .max_depth = 10,
    },
    .preview => .{
        .image_height = 225,
        .samples_per_pixel = 100,
        .max_depth = 50,
    },
    .fhd => .{
        .image_height = 1080,
        .samples_per_pixel = 500,
        .max_depth = 50,
    },
};

fn renderSceneToPng(world: *const World, file_name: [:0]const u8) !void {
    var fb = Framebuffer.init(cfg.image_height, world.camera.aspect_ratio);
    defer fb.deinit();

    var queue = RenderChunkQueue.init();
    defer queue.deinit();

    var line: usize = 0;
    while (line < fb.height) : (line += scanlines_per_chunk) {
        try queue.chunks.append(.{
            .rand = Rand.init(global_rand.uniformInt64()),
            .y_start = line,
            .y_end = std.math.min(line + scanlines_per_chunk, fb.height),
        });
    }

    // Run threads.
    var threads: [num_threads]std.Thread = undefined;

    for (threads) |_, i| {
        threads[i] = try std.Thread.spawn(.{}, renderThread, .{ fb, world, &queue });
    }

    for (threads) |thread| {
        thread.join();
    }

    stb.stbi_flip_vertically_on_write(1);
    _ = stb.stbi_write_png(
        file_name,
        @intCast(c_int, fb.width),
        @intCast(c_int, fb.height),
        3,
        @ptrCast(*const anyopaque, fb.image.items),
        @intCast(c_int, fb.width * 3),
    );
}

pub fn main() !void {
    // TODO: Does zig not have anything to reflect the name of a function?
    const SceneRender = struct {
        fun: fn () World,
        file_name: [:0]const u8,
    };
    const scenes = [_]SceneRender{
        .{ .fun = sceneInitialGrey, .file_name = "01_initial_gray.png" },
        .{ .fun = sceneFuzzedMetal, .file_name = "02_fuzzed_metal.png" },
        .{ .fun = sceneDieletricHollow, .file_name = "03_dielectric_hollow.png" },
        .{ .fun = sceneDieletricHollowNewPerspective, .file_name = "04_dielectric_hollow_new_perspective.png" },
        .{ .fun = sceneBook1FinalRandom, .file_name = "05_book1_final_random.png" },
    };

    for (scenes) |scene| {
        std.debug.print("\nRender scene: {s}\n", .{scene.file_name});

        var world = scene.fun();
        defer world.deinit();

        try renderSceneToPng(&world, scene.file_name);
    }
}

fn sceneInitialGrey() World {
    const cam = Camera.init(zero, -unit_z, unit_y, 90.0, (16.0 / 9.0), 0.0, 1.0);
    var world = World.init(cam);

    const mat_gray = world.addMaterial(.{
        .lambertian = .{ .albedo = scalar(0.5) },
    });

    world.addSphere(.{
        .center = .{ 0.0, -100.5, -1.0 },
        .radius = 100.0,
        .material = mat_gray,
    });
    world.addSphere(.{
        .center = -unit_z,
        .radius = 0.5,
        .material = mat_gray,
    });

    return world;
}

fn sceneFuzzedMetal() World {
    const cam = Camera.init(zero, -unit_z, unit_y, 90.0, (16.0 / 9.0), 0.0, 1.0);
    var world = World.init(cam);

    const mat_ground = world.addMaterial(.{
        .lambertian = .{ .albedo = .{ 0.8, 0.8, 0.0 } },
    });
    const mat_center = world.addMaterial(.{
        .lambertian = .{ .albedo = .{ 0.7, 0.3, 0.3 } },
    });
    const mat_left = world.addMaterial(.{
        .metal = .{ .albedo = .{ 0.8, 0.8, 0.8 }, .fuzz = 0.3 },
    });
    const mat_right = world.addMaterial(.{
        .metal = .{ .albedo = .{ 0.8, 0.6, 0.2 }, .fuzz = 1.0 },
    });

    world.addSphere(.{
        .center = .{ 0.0, -100.5, -1.0 },
        .radius = 100.0,
        .material = mat_ground,
    });
    world.addSphere(.{
        .center = -unit_z,
        .radius = 0.5,
        .material = mat_center,
    });
    world.addSphere(.{
        .center = -unit_z - unit_x,
        .radius = 0.5,
        .material = mat_left,
    });
    world.addSphere(.{
        .center = -unit_z + unit_x,
        .radius = 0.5,
        .material = mat_right,
    });

    return world;
}

fn sceneDieletricHollow() World {
    const cam = Camera.init(zero, -unit_z, unit_y, 90.0, (16.0 / 9.0), 0.0, 1.0);
    var world = World.init(cam);

    const mat_ground = world.addMaterial(.{
        .lambertian = .{ .albedo = .{ 0.8, 0.8, 0.0 } },
    });
    const mat_center = world.addMaterial(.{
        .lambertian = .{ .albedo = .{ 0.1, 0.2, 0.5 } },
    });
    const mat_left = world.addMaterial(.{
        .dielectric = .{ .refraction_index = 1.5 },
    });
    const mat_right = world.addMaterial(.{
        .metal = .{ .albedo = .{ 0.8, 0.6, 0.2 }, .fuzz = 0.0 },
    });

    world.addSphere(.{
        .center = .{ 0.0, -100.5, -1.0 },
        .radius = 100.0,
        .material = mat_ground,
    });
    world.addSphere(.{
        .center = -unit_z,
        .radius = 0.5,
        .material = mat_center,
    });
    world.addSphere(.{
        .center = -unit_z - unit_x,
        .radius = 0.5,
        .material = mat_left,
    });
    world.addSphere(.{
        .center = -unit_z - unit_x,
        .radius = -0.4,
        .material = mat_left,
    });
    world.addSphere(.{
        .center = -unit_z + unit_x,
        .radius = 0.5,
        .material = mat_right,
    });

    return world;
}

fn sceneDieletricHollowNewPerspective() World {
    var v = sceneDieletricHollow();
    v.camera = Camera.init(.{ -2.0, 2.0, 1.0 }, -unit_z, unit_y, 20.0, (16.0 / 9.0), 0.0, 1.0);
    return v;
}

fn sceneDepthOfField() World {
    var v = sceneDieletricHollow();
    const from = Vec3{ 3.0, 3.0, 2.0 };
    const to = -unit_z;
    const dist_to_focus = length(to - from);
    const aperture = 2.0;
    v.camera = Camera.init(from, to, unit_y, 20.0, (16.0 / 9.0), aperture, dist_to_focus);
    return v;
}

fn sceneBook1FinalRandom() World {
    const cam = Camera.init(.{ 13.0, 2.0, 3.0 }, zero, unit_y, 20.0, (3.0 / 2.0), 0.1, 10.0);
    var world = World.init(cam);

    const mat_gray = world.addMaterial(.{
        .lambertian = .{ .albedo = scalar(0.5) },
    });
    world.addSphere(.{
        .center = .{ 0.0, -1000.0, -1.0 },
        .radius = 1000.0,
        .material = mat_gray,
    });

    const mat1 = world.addMaterial(.{ .dielectric = .{ .refraction_index = 1.5 } });
    world.addSphere(.{ .center = .{ 0.0, 1.0, 0.0 }, .radius = 1.0, .material = mat1 });

    const mat2 = world.addMaterial(.{ .lambertian = .{ .albedo = .{ 0.4, 0.2, 0.1 } } });
    world.addSphere(.{ .center = .{ -4.0, 1.0, 0.0 }, .radius = 1.0, .material = mat2 });

    const mat3 = world.addMaterial(.{ .metal = .{ .albedo = .{ 0.7, 0.6, 0.5 }, .fuzz = 0.0 } });
    world.addSphere(.{ .center = .{ 4.0, 1.0, 0.0 }, .radius = 1.0, .material = mat3 });

    const mat_dielectric = mat1;
    const radius = 0.2;

    var a: isize = -11;
    while (a < 11) : (a += 1) {
        var b: isize = -11;
        while (b < 11) : (b += 1) {
            const center = Vec3{ @intToFloat(f32, a) + 0.9 * global_rand.uniformFloat(), 0.2, @intToFloat(f32, b) + 0.9 * global_rand.uniformFloat() };

            if (length(center - Vec3{ 4, 0.2, 0 }) > 0.9) {
                const choose_mat = global_rand.uniformFloat();

                if (choose_mat < 0.8) { // diffuse
                    const mat = world.addMaterial(.{
                        .lambertian = .{
                            .albedo = global_rand.uniformVector() * global_rand.uniformVector(),
                        },
                    });
                    world.addSphere(.{ .center = center, .radius = radius, .material = mat });
                } else if (choose_mat < 0.95) { // metal
                    const mat = world.addMaterial(.{
                        .metal = .{
                            .albedo = global_rand.uniformVectorInRange(0.5, 1.0),
                            .fuzz = global_rand.uniformFloatInRange(0, 0.5),
                        },
                    });
                    world.addSphere(.{ .center = center, .radius = radius, .material = mat });
                } else { // glass
                    world.addSphere(.{ .center = center, .radius = radius, .material = mat_dielectric });
                }
            }
        }
    }

    return world;
}

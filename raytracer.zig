const std = @import("std");

const stb = @cImport({
    @cUndef("STB_IMAGE_WRITE_IMPLEMENTATION");
    @cInclude("stb_image_write.c");
});

var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
const gpa = general_purpose_allocator.allocator();

var pcg = std.rand.Pcg.init(0x853c49e6748fea9b);

const pi = std.math.pi;
fn deg2Rad(deg: f32) f32 {
    return deg / 180.0 * pi;
    // In Zig 10.0
    // return std.math.degreesToRadians(f32, deg);
}
const infinity = std.math.inf(f32);
fn randFloat() f32 {
    return pcg.random().float(f32);
}
fn randFloatRange(min: f32, max: f32) f32 {
    return min + (max - min) * randFloat();
}

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

const RGB8 = struct {
    r: u8,
    g: u8,
    b: u8,

    pub fn init(color_acc: Color, samples_per_pixel: usize) RGB8 {
        const color_avg = color_acc / scalar(@intToFloat(f32, samples_per_pixel));
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
        rec.material = self.material;
        const outward_normal = (rec.pos - self.center) / scalar(self.radius);
        rec.setFaceNormal(r, outward_normal);
        return rec;
    }
};

const World = struct {
    spheres: std.ArrayList(Sphere),
    materials: std.ArrayList(Material),

    pub fn init() World {
        return .{
            .spheres = std.ArrayList(Sphere).init(gpa),
            .materials = std.ArrayList(Material).init(gpa),
        };
    }
    pub fn deinit(self: World) void {
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

    pub fn init(look_from: Vec3, look_at: Vec3, up: Vec3, fovy: f32, aspect_ratio: f32) Camera {
        const theta = deg2Rad(fovy);
        const h = std.math.tan(theta * 0.5);
        const viewport_height = 2.0 * h;
        const viewport_width = aspect_ratio * viewport_height;

        const w = normalize(look_from - look_at);
        const u = normalize(cross(up, w));
        const v = cross(w, u);

        var res: Camera = undefined;
        res.origin = look_from;
        res.horizontal = u * scalar(viewport_width);
        res.vertical = v * scalar(viewport_height);
        res.lower_left_corner = res.origin - scalar(0.5) * res.horizontal - scalar(0.5) * res.vertical - w;
        return res;
    }

    pub fn getRay(self: Camera, s: f32, t: f32) Ray {
        const pixel_pos = self.lower_left_corner + scalar(s) * self.horizontal + scalar(t) * self.vertical;
        return Ray{
            .origin = self.origin,
            .dir = pixel_pos - self.origin,
        };
    }
};

// Archimedes' Hat-Box Theorem
fn randomUnitVector() Vec3 {
    const w_z = 2.0 * randFloat() - 1.0;
    const r = @sqrt(1.0 - w_z * w_z);
    const theta = 2.0 * pi * randFloat();
    const w_x = r * @cos(theta);
    const w_y = r * @sin(theta);
    return .{ w_x, w_y, w_z };
}
fn randomInUnitSphereRejectionSampling() Vec3 {
    return while (true) {
        const p = Vec3{
            randFloatRange(-1.0, 1.0),
            randFloatRange(-1.0, 1.0),
            randFloatRange(-1.0, 1.0),
        };
        if (lengthSquared(p) < 1.0) {
            break p;
        }
    } else unreachable;
}
// Is it correct to use the cubic root of a uniform random number as radius?
// Probably: https://stackoverflow.com/a/5408843
fn randomInUnitSphere() Vec3 {
    const rand_unit = randomUnitVector();
    const radius = std.math.pow(f32, randFloat(), 1.0 / 3.0);
    return scalar(radius) * rand_unit;
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

const Scatter = struct {
    ray_out: Ray,
    attenuation: Color,
};

const Lambertian = struct {
    albedo: Vec3,

    pub fn scatter(self: Lambertian, hit: HitRecord) Scatter {
        // This is cosine weighted importance sampling of the direction.
        // (See: https://twitter.com/mmalex/status/1550765798263758848)
        // The pdf is cos(theta) / pi
        // cos cancels the cos in the lighting equation
        // A proper Lambertian BRDF is albeda / pi
        // The pis in the pdf and the BRDF cancel each other.
        var scatter_dir = hit.normal + randomUnitVector();
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

    pub fn scatter(self: Metal, r_in: Ray, hit: HitRecord) ?Scatter {
        // NOTE: In the book, r_in.dir is normalized. I don't think that is necessary.
        const reflected = reflect(r_in.dir, hit.normal);
        const dir_out = reflected + scalar(self.fuzz) * randomInUnitSphere();

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

    pub fn scatter(self: Dielectric, r_in: Ray, hit: HitRecord) Scatter {
        const refraction_ratio = if (hit.is_front_face) (1.0 / self.refraction_index) else self.refraction_index;
        const unit_dir = normalize(r_in.dir);

        // TODO: cos_theta is also computed inside refract(...), only do it once.
        const cos_theta = std.math.min(dot(-unit_dir, hit.normal), 1.0);
        const sin_theta = @sqrt(1.0 - cos_theta * cos_theta);
        const cannot_refract = refraction_ratio * sin_theta > 1.0;

        // Read "Reflections and Refractions in Ray Tracing" about TIR etc.
        var out_dir: Vec3 = undefined;
        if (cannot_refract or reflectanceSchlickApprox(cos_theta, self.refraction_index) > randFloat()) {
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

    pub fn scatter(self: Material, r_in: Ray, hit: HitRecord) ?Scatter {
        return switch (self) {
            .lambertian => |l| l.scatter(hit),
            .metal => |m| m.scatter(r_in, hit),
            .dielectric => |d| d.scatter(r_in, hit),
        };
    }
};
const MaterialHandle = u16;

fn rayColor(world: *World, r: Ray, depth: isize) Color {
    if (depth <= 0) {
        return black;
    }

    const intersection = world.hit(r, 0.001, infinity);
    if (intersection) |hit| {
        const mat = world.materials.items[hit.material];
        if (mat.scatter(r, hit)) |scatter| {
            return scatter.attenuation * rayColor(world, scatter.ray_out, depth - 1);
        } else {
            return black;
        }
    } else {
        const t = scalar(0.5 * (normalize(r.dir)[1] + 1.0));
        return (one - t) * white + t * Color{ 0.5, 0.7, 1.0 };
    }
}

pub fn main() !void {
    const aspect_ratio = 16.0 / 9.0;

    const image_width = 400;
    const image_height = @floatToInt(usize, @as(f32, image_width) / aspect_ratio);
    var image: [image_height][image_width]RGB8 = undefined;
    const samples_per_pixel = 100;
    const max_depth = 50;

    const cam = Camera.init(zero, -unit_z, unit_y, 90.0, aspect_ratio);
    //const cam = Camera.init(.{ -2.0, 2.0, 1.0 }, -unit_z, unit_y, 20.0, aspect_ratio);

    //var world = sceneInitialGrey();
    //var world = sceneFuzzedMetal();
    var world = sceneDieletricHollow();
    defer world.deinit();

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
                pixel_color += rayColor(&world, ray, max_depth);
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

fn sceneInitialGrey() World {
    var world = World.init();

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
    var world = World.init();

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
    var world = World.init();

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

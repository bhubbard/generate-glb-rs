use crate::error::{GenerateGlbError, Result};
use serde::{Deserialize, Serialize};

/// 3D Mesh structure representing vertices, normals, RGBA colors, and indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn len_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn len_faces(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    /// Parses an OBJ format string into a Mesh.
    pub fn from_obj_string(obj_str: &str) -> Result<Self> {
        let mut raw_vertices = Vec::new();
        let mut mesh = Self::new();

        for line in obj_str.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut tokens = trimmed.split_whitespace();
            let tag = match tokens.next() {
                Some(t) => t,
                None => continue,
            };

            match tag {
                "v" => {
                    let x: f32 = tokens.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let y: f32 = tokens.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let z: f32 = tokens.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    raw_vertices.push([x, y, z]);
                }
                "f" => {
                    let face_tokens: Vec<&str> = tokens.collect();
                    if face_tokens.len() < 3 {
                        continue;
                    }

                    // Polygon fan triangulation
                    for i in 1..face_tokens.len() - 1 {
                        let i0 = parse_face_index(face_tokens[0]);
                        let i1 = parse_face_index(face_tokens[i]);
                        let i2 = parse_face_index(face_tokens[i + 1]);

                        if let (Some(idx0), Some(idx1), Some(idx2)) = (i0, i1, i2) {
                            if idx0 > 0 && idx1 > 0 && idx2 > 0 {
                                let v0 = (idx0 - 1) as usize;
                                let v1 = (idx1 - 1) as usize;
                                let v2 = (idx2 - 1) as usize;

                                if v0 < raw_vertices.len() && v1 < raw_vertices.len() && v2 < raw_vertices.len() {
                                    let new_base = mesh.vertices.len() as u32;
                                    mesh.vertices.push(raw_vertices[v0]);
                                    mesh.vertices.push(raw_vertices[v1]);
                                    mesh.vertices.push(raw_vertices[v2]);

                                    mesh.indices.push(new_base);
                                    mesh.indices.push(new_base + 1);
                                    mesh.indices.push(new_base + 2);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if mesh.vertices.is_empty() {
            return Err(GenerateGlbError::InvalidMesh(
                "No valid vertices or faces found in OBJ data".to_string(),
            ));
        }

        mesh.compute_normals();
        mesh.apply_gradient_color();
        Ok(mesh)
    }

    /// Computes smooth vertex normals by accumulating weighted face normals.
    pub fn compute_normals(&mut self) {
        let num_verts = self.vertices.len();
        let mut accum_normals = vec![[0.0f32; 3]; num_verts];

        for chunk in self.indices.chunks_exact(3) {
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            let v0 = self.vertices[i0];
            let v1 = self.vertices[i1];
            let v2 = self.vertices[i2];

            // Edge vectors
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

            // Cross product
            let nx = e1[1] * e2[2] - e1[2] * e2[1];
            let ny = e1[2] * e2[0] - e1[0] * e2[2];
            let nz = e1[0] * e2[1] - e1[1] * e2[0];

            accum_normals[i0][0] += nx; accum_normals[i0][1] += ny; accum_normals[i0][2] += nz;
            accum_normals[i1][0] += nx; accum_normals[i1][1] += ny; accum_normals[i1][2] += nz;
            accum_normals[i2][0] += nx; accum_normals[i2][1] += ny; accum_normals[i2][2] += nz;
        }

        self.normals = accum_normals
            .into_iter()
            .map(|n| {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 1e-6 {
                    [n[0] / len, n[1] / len, n[2] / len]
                } else {
                    [0.0, 1.0, 0.0]
                }
            })
            .collect();
    }

    /// Applies a vertical Y-axis color gradient matching upstream `generate.py`:
    /// Red = normalized_y, Blue = 1 - normalized_y, Alpha = 1.0.
    pub fn apply_gradient_color(&mut self) {
        if self.vertices.is_empty() {
            return;
        }

        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;

        for v in &self.vertices {
            y_min = y_min.min(v[1]);
            y_max = y_max.max(v[1]);
        }

        let y_range = if (y_max - y_min) > 1e-6 {
            y_max - y_min
        } else {
            1.0
        };

        self.colors = self
            .vertices
            .iter()
            .map(|v| {
                let y_norm = if (y_max - y_min) > 1e-6 {
                    ((v[1] - y_min) / y_range).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                [
                    y_norm,         // Red channel
                    0.0,            // Green channel
                    1.0 - y_norm,   // Blue channel
                    1.0,            // Alpha channel
                ]
            })
            .collect();
    }

    /// Computes the bounding box of the mesh: ([min_x, min_y, min_z], [max_x, max_y, max_z]).
    pub fn bounding_box(&self) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];

        for v in &self.vertices {
            min[0] = min[0].min(v[0]);
            min[1] = min[1].min(v[1]);
            min[2] = min[2].min(v[2]);

            max[0] = max[0].max(v[0]);
            max[1] = max[1].max(v[1]);
            max[2] = max[2].max(v[2]);
        }

        (min, max)
    }
}

fn parse_face_index(token: &str) -> Option<u32> {
    let clean = token.split('/').next()?;
    clean.parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_obj() {
        let obj = r#"
        # Simple Pyramid
        v 0.0 1.0 0.0
        v -1.0 0.0 -1.0
        v 1.0 0.0 -1.0
        v 1.0 0.0 1.0
        v -1.0 0.0 1.0
        f 1 2 3
        f 1 3 4
        f 1 4 5
        f 1 5 2
        f 2 4 3
        f 2 5 4
        "#;

        let mesh = Mesh::from_obj_string(obj).unwrap();
        assert_eq!(mesh.len_faces(), 6);
        assert_eq!(mesh.vertices.len(), 18); // 6 triangles * 3
        assert_eq!(mesh.normals.len(), 18);
        assert_eq!(mesh.colors.len(), 18);

        // Top apex (y = 1.0) should be pure Red ([1.0, 0.0, 0.0, 1.0])
        // Base (y = 0.0) should be pure Blue ([0.0, 0.0, 1.0, 1.0])
        let c0 = mesh.colors[0];
        assert!((c0[0] - 1.0).abs() < 1e-4);
        assert!((c0[2] - 0.0).abs() < 1e-4);
    }
}

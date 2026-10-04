use crate::error::Result;
use crate::mesh::Mesh;
use byteorder::{LittleEndian, WriteBytesExt};
use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Exports a 3D Mesh to a binary glTF 2.0 (.glb) format bytes or file.
pub fn export_glb_bytes(mesh: &Mesh) -> Result<Vec<u8>> {
    let (min_pos, max_pos) = mesh.bounding_box();
    let num_verts = mesh.vertices.len();
    let num_indices = mesh.indices.len();

    let mut bin_data = Vec::new();

    // 1. Pack Positions (VEC3, FLOAT 5126)
    let pos_offset = bin_data.len();
    for v in &mesh.vertices {
        bin_data.write_f32::<LittleEndian>(v[0])?;
        bin_data.write_f32::<LittleEndian>(v[1])?;
        bin_data.write_f32::<LittleEndian>(v[2])?;
    }
    let pos_length = bin_data.len() - pos_offset;
    pad_to_4_bytes(&mut bin_data);

    // 2. Pack Normals (VEC3, FLOAT 5126)
    let norm_offset = bin_data.len();
    for n in &mesh.normals {
        bin_data.write_f32::<LittleEndian>(n[0])?;
        bin_data.write_f32::<LittleEndian>(n[1])?;
        bin_data.write_f32::<LittleEndian>(n[2])?;
    }
    let norm_length = bin_data.len() - norm_offset;
    pad_to_4_bytes(&mut bin_data);

    // 3. Pack Colors (VEC4, FLOAT 5126)
    let color_offset = bin_data.len();
    for c in &mesh.colors {
        bin_data.write_f32::<LittleEndian>(c[0])?;
        bin_data.write_f32::<LittleEndian>(c[1])?;
        bin_data.write_f32::<LittleEndian>(c[2])?;
        bin_data.write_f32::<LittleEndian>(c[3])?;
    }
    let color_length = bin_data.len() - color_offset;
    pad_to_4_bytes(&mut bin_data);

    // 4. Pack Indices (SCALAR, UNSIGNED_INT 5125)
    let indices_offset = bin_data.len();
    for &idx in &mesh.indices {
        bin_data.write_u32::<LittleEndian>(idx)?;
    }
    let indices_length = bin_data.len() - indices_offset;
    pad_to_4_bytes(&mut bin_data);

    // Build glTF JSON metadata
    let gltf_json = json!({
        "asset": {
            "version": "2.0",
            "generator": "generate-glb-rs (LLaMA-Mesh)"
        },
        "scene": 0,
        "scenes": [
            { "name": "Scene", "nodes": [0] }
        ],
        "nodes": [
            { "name": "LLaMA_Mesh", "mesh": 0 }
        ],
        "meshes": [
            {
                "name": "GeneratedModel",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "COLOR_0": 2
                        },
                        "indices": 3,
                        "material": 0
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "VertexColorMaterial",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.15,
                    "roughnessFactor": 0.6
                },
                "doubleSided": true
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126, // FLOAT
                "count": num_verts,
                "type": "VEC3",
                "min": min_pos,
                "max": max_pos
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": num_verts,
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126,
                "count": num_verts,
                "type": "VEC4"
            },
            {
                "bufferView": 3,
                "byteOffset": 0,
                "componentType": 5125, // UNSIGNED_INT
                "count": num_indices,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": pos_length,
                "target": 34962 // ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": norm_offset,
                "byteLength": norm_length,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": color_offset,
                "byteLength": color_length,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": indices_offset,
                "byteLength": indices_length,
                "target": 34963 // ELEMENT_ARRAY_BUFFER
            }
        ],
        "buffers": [
            {
                "byteLength": bin_data.len()
            }
        ]
    });

    let json_string = serde_json::to_string(&gltf_json)?;
    let mut json_bytes = json_string.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' '); // Pad JSON with spaces
    }

    let total_length = 12 + 8 + json_bytes.len() + 8 + bin_data.len();
    let mut glb_output = Vec::with_capacity(total_length);

    // 12-byte header
    glb_output.write_u32::<LittleEndian>(0x46546C67)?; // Magic: 'glTF'
    glb_output.write_u32::<LittleEndian>(2)?;          // Version: 2
    glb_output.write_u32::<LittleEndian>(total_length as u32)?;

    // Chunk 0: JSON
    glb_output.write_u32::<LittleEndian>(json_bytes.len() as u32)?;
    glb_output.write_u32::<LittleEndian>(0x4E4F534A)?; // 'JSON'
    glb_output.extend_from_slice(&json_bytes);

    // Chunk 1: BIN
    glb_output.write_u32::<LittleEndian>(bin_data.len() as u32)?;
    glb_output.write_u32::<LittleEndian>(0x004E4942)?; // 'BIN\0'
    glb_output.extend_from_slice(&bin_data);

    Ok(glb_output)
}

/// Saves a 3D Mesh directly to a .glb file on disk.
pub fn save_glb<P: AsRef<Path>>(mesh: &Mesh, path: P) -> Result<()> {
    let bytes = export_glb_bytes(mesh)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}

fn pad_to_4_bytes(vec: &mut Vec<u8>) {
    while vec.len() % 4 != 0 {
        vec.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_export_glb() {
        let obj = r#"
        v 0.0 1.0 0.0
        v -1.0 0.0 0.0
        v 1.0 0.0 0.0
        f 1 2 3
        "#;
        let mesh = Mesh::from_obj_string(obj).unwrap();
        let bytes = export_glb_bytes(&mesh).unwrap();

        assert_eq!(&bytes[0..4], b"glTF");
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 2);

        let tmp = NamedTempFile::new().unwrap();
        save_glb(&mesh, tmp.path()).unwrap();
        let file_bytes = std::fs::read(tmp.path()).unwrap();
        assert_eq!(bytes, file_bytes);
    }
}

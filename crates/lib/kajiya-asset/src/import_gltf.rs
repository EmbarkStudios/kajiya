// Based on `import.rs` in the `gltf` crate, but modified not to load images (we do that separately).

use bytes::Bytes;
use gltf::{buffer, image, Document, Error, Gltf, Result};
use std::{fs, io, path::Path};

use crate::image::ImageSource;

type BufferBytes = Bytes;

/// Return type of `import`.
type Import = (Document, Vec<BufferBytes>, Vec<ImageSource>);

/// Represents the set of URI schemes the importer supports.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Scheme<'a> {
    /// `data:[<media type>];base64,<data>`.
    Data(Option<&'a str>, &'a str),

    /// `file:[//]<absolute file path>`.
    ///
    /// Note: The file scheme does not implement authority.
    File(&'a str),

    /// `../foo`, etc.
    Relative,

    /// Placeholder for an unsupported URI scheme identifier.
    Unsupported,
}

impl<'a> Scheme<'a> {
    fn parse(uri: &str) -> Scheme<'_> {
        if uri.contains(':') {
            #[allow(clippy::manual_strip)]
            #[allow(clippy::iter_nth_zero)]
            if uri.starts_with("data:") {
                let match0 = &uri["data:".len()..].split(";base64,").nth(0);
                let match1 = &uri["data:".len()..].split(";base64,").nth(1);
                if match1.is_some() {
                    Scheme::Data(Some(match0.unwrap()), match1.unwrap())
                } else if match0.is_some() {
                    Scheme::Data(None, match0.unwrap())
                } else {
                    Scheme::Unsupported
                }
            } else if uri.starts_with("file://") {
                Scheme::File(&uri["file://".len()..])
            } else if uri.starts_with("file:") {
                Scheme::File(&uri["file:".len()..])
            } else {
                Scheme::Unsupported
            }
        } else {
            Scheme::Relative
        }
    }

    fn read(base: Option<&Path>, uri: &str) -> Result<Vec<u8>> {
        match Scheme::parse(uri) {
            // The path may be unused in the Scheme::Data case
            // Example: "uri" : "data:application/octet-stream;base64,wsVHPgA...."
            Scheme::Data(_, base64) => base64::decode(&base64).map_err(Error::Base64),
            Scheme::File(path) if base.is_some() => read_to_end(path),
            Scheme::Relative if base.is_some() => read_to_end(base.unwrap().join(uri)),
            Scheme::Unsupported => Err(Error::UnsupportedScheme),
            _ => Err(Error::ExternalReferenceInSliceImport),
        }
    }
}

fn read_to_end<P>(path: P) -> Result<Vec<u8>>
where
    P: AsRef<Path>,
{
    use io::Read;
    let file = fs::File::open(path.as_ref()).map_err(Error::Io)?;
    // Allocate one extra byte so the buffer doesn't need to grow before the
    // final `read` call at the end of the file.  Don't worry about `usize`
    // overflow because reading will fail regardless in that case.
    let length = file.metadata().map(|x| x.len() + 1).unwrap_or(0);
    let mut reader = io::BufReader::new(file);
    let mut data = Vec::with_capacity(length as usize);
    reader.read_to_end(&mut data).map_err(Error::Io)?;
    Ok(data)
}

/// Import the buffer data referenced by a glTF document.
pub fn import_buffer_data(
    document: &Document,
    base: Option<&Path>,
    mut blob: Option<Vec<u8>>,
) -> Result<Vec<Bytes>> {
    let mut buffers = Vec::new();
    for buffer in document.buffers() {
        let mut data = match buffer.source() {
            buffer::Source::Uri(uri) => Scheme::read(base, uri),
            buffer::Source::Bin => blob.take().ok_or(Error::MissingBlob),
        }?;
        if data.len() < buffer.length() {
            return Err(Error::BufferLength {
                buffer: buffer.index(),
                expected: buffer.length(),
                actual: data.len(),
            });
        }
        while data.len() % 4 != 0 {
            data.push(0);
        }
        buffers.push(Bytes::from(data));
    }
    Ok(buffers)
}

/// Import the image data referenced by a glTF document.
pub fn import_image_data(
    document: &Document,
    base: Option<&Path>,
    buffer_data: &[Bytes],
) -> Result<Vec<ImageSource>> {
    let mut images = Vec::new();

    for image in document.images() {
        match image.source() {
            image::Source::Uri { uri, mime_type: _ } if base.is_some() => {
                let uri = urlencoding::decode(uri).map_err(|_| Error::UnsupportedScheme)?;
                let uri = uri.as_ref();

                match Scheme::parse(uri) {
                    Scheme::Data(Some(_mime_type), base64) => {
                        let bytes = base64::decode(&base64).map_err(Error::Base64)?;
                        images.push(ImageSource::Memory(Bytes::from(bytes)));
                    }
                    Scheme::Data(None, ..) => return Err(Error::ExternalReferenceInSliceImport),
                    Scheme::Unsupported => return Err(Error::UnsupportedScheme),
                    Scheme::File(path) => images.push(ImageSource::File(path.into())),
                    Scheme::Relative if base.is_some() => {
                        images.push(ImageSource::File(base.unwrap().join(uri)))
                    }
                    Scheme::Relative => return Err(Error::UnsupportedScheme),
                }
            }
            image::Source::View { view, mime_type: _ } => {
                let parent_buffer_data = &buffer_data[view.buffer().index()];
                let begin = view.offset();
                let end = begin + view.length();
                let encoded_image = parent_buffer_data.slice(begin..end);
                images.push(ImageSource::Memory(encoded_image));
            }
            _ => return Err(Error::ExternalReferenceInSliceImport),
        }
    }

    Ok(images)
}

fn import_impl(Gltf { document, blob }: Gltf, base: Option<&Path>) -> Result<Import> {
    let buffer_data = import_buffer_data(&document, base, blob)?;
    let image_data = import_image_data(&document, base, &buffer_data)?;
    let import = (document, buffer_data, image_data);
    Ok(import)
}

fn import_path(path: &Path) -> Result<Import> {
    let base = path.parent().unwrap_or_else(|| Path::new("./"));
    let file = fs::File::open(path).map_err(Error::Io)?;
    let reader = io::BufReader::new(file);
    import_impl(Gltf::from_reader(reader)?, Some(base))
}

/// Import some glTF 2.0 from the file system.
pub fn import<P>(path: P) -> Result<Import>
where
    P: AsRef<Path>,
{
    import_path(path.as_ref())
}

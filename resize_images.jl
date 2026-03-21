using FileIO
using Images
using ImageTransformations

# 1. Configure input/output paths
data_path = get(ENV, "SPLATLAB_DATASET_PATH", "/path/to/colmap/dataset")
src_dir = joinpath(data_path, "images")
dst_dir = joinpath(data_path, "images_4")

# 2. Create output directory if it does not exist
mkpath(dst_dir)

# 3. Find all image files
files = filter(x -> endswith(lowercase(x), ".jpg") || endswith(lowercase(x), ".png"), readdir(src_dir))

println("Starting image resize, found $(length(files)) images...")

# 4. Process each image
for (i, file) in enumerate(files)
    # Load source image
    img = load(joinpath(src_dir, file))
    
    # Compute size scaled by 4
    new_size = (size(img, 1) ÷ 4, size(img, 2) ÷ 4)
    
    # Resize
    img_resized = imresize(img, new_size)
    
    # Save to output folder
    save(joinpath(dst_dir, file), img_resized)
    
    if i % 50 == 0 || i == length(files)
        println("Processed $i / $(length(files))")
    end
end

println("Done. images_4 has been generated.")
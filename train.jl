import GaussianSplatting as GSP
using CUDA
using CSV
using DataFrames
using FileIO
using ImageIO
using ColorTypes

# 1. Basic settings
data_path = get(ENV, "SPLATLAB_DATASET_PATH", "/path/to/colmap/dataset")
kab = CUDABackend()

println("Loading dataset (scale=4 to reduce VRAM usage)...")
dataset = GSP.ColmapDataset(kab, data_path; scale=4, train_test_split=0.9, permute=true)

println("Initializing Gaussian model and rasterizer...")
gaussians = GSP.GaussianModel(dataset.points, dataset.colors, dataset.scales; max_sh_degree=1)
rasterizer = GSP.GaussianRasterizer(kab; width=496, height=272, mode=:rgb)
opt_params = GSP.OptimizationParams()
trainer = GSP.Trainer(rasterizer, gaussians, dataset, opt_params)

# 3. Containers for metrics logging
iterations = 3000 
history_loss_100 = Float64[]  # Log loss every 100 steps to align with SSIM/PSNR plots
history_ssim = Float64[]
history_psnr = Float64[]
history_num_gaussians = Int[] # Track Gaussian count growth

# Select a fixed camera view for progress snapshots
test_camera = dataset.train_cameras[10]

# 4. Custom training loop
println("Starting custom training, total iterations: $iterations")
for i in 1:iterations
    loss = GSP.step!(trainer)
    
    # Print progress and collect plot data every 100 steps
    if i % 100 == 0 || i == iterations
        metrics = GSP.validate(trainer)
        
        # Count current Gaussians (opacities length equals Gaussian count)
        current_num_gaussians = length(gaussians.opacities)
        
        # Append metrics
        push!(history_loss_100, loss)
        push!(history_ssim, metrics.eval_ssim)
        push!(history_psnr, metrics.eval_psnr)
        push!(history_num_gaussians, current_num_gaussians)
        
        println("[Iteration $i/$iterations] Loss: $(round(loss, digits=4)) | SSIM: $(round(metrics.eval_ssim, digits=4)) | PSNR: $(round(metrics.eval_psnr, digits=2)) | Gaussian Count: $current_num_gaussians")
        
        # Save intermediate snapshots at key milestones
        if i in [100, 1000, iterations]
            println("Rendering snapshot at step $i...")
            image_features = rasterizer(
                gaussians.points, gaussians.opacities, gaussians.scales,
                gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
                camera=test_camera, sh_degree=gaussians.sh_degree
            )
            host_image_features = Array(image_features)
            img_to_save = GSP.to_image(@view(host_image_features[1:3, :, :]))
            save("evolution_step_$i.png", img_to_save)
        end
    end
end

println("Training completed. All snapshots have been saved.")

# 5. Export CSV metrics
df = DataFrame(
    Iteration = 100:100:iterations, 
    Loss = history_loss_100,
    SSIM = history_ssim, 
    PSNR = history_psnr,
    Num_Gaussians = history_num_gaussians
)
CSV.write("training_metrics.csv", df)
println("Validation metrics saved to training_metrics.csv")
using CSV
using DataFrames
using Plots

# Global plotting style for report-quality figures
default(linewidth=2.5, marker=:circle, markersize=4, grid=true, gridalpha=0.3, framestyle=:box, fontfamily="sans-serif")

println("Reading training metrics...")
df = CSV.read("training_metrics.csv", DataFrame)

# 1. Loss convergence curve
p1 = plot(df.Iteration, df.Loss, 
    title="Training Loss Convergence", xlabel="Iterations", ylabel="L1 Loss", 
    color=:crimson, legend=false)
savefig(p1, "chart_1_loss.png")

# 2. Gaussian count growth curve
p2 = plot(df.Iteration, df.Num_Gaussians, 
    title="Growth of Gaussian Splats", xlabel="Iterations", ylabel="Number of Gaussians", 
    color=:purple, marker=:square, legend=false)
savefig(p2, "chart_2_gaussians_growth.png")

# 3. SSIM quality curve
p3 = plot(df.Iteration, df.SSIM, 
    title="SSIM Quality Assessment", xlabel="Iterations", ylabel="SSIM Score", 
    color=:royalblue, legend=false)
savefig(p3, "chart_3_ssim.png")

# 4. PSNR quality curve
p4 = plot(df.Iteration, df.PSNR, 
    title="PSNR Quality Assessment", xlabel="Iterations", ylabel="PSNR (dB)", 
    color=:forestgreen, legend=false)
savefig(p4, "chart_4_psnr.png")

println("All report charts generated successfully.")
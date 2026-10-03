package gpu

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -I../../gpubrain -DMAX_VALUES=1024u -DMAX_MEMORY=256u -DMAX_PLASTIC=256u -DLEVEL_EDGES=520u -DLANES=64u netrun.comp -o netrun.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -I../../gpubrain -DMAX_VALUES=2048u -DMAX_MEMORY=1024u -DMAX_PLASTIC=1024u -DLEVEL_EDGES=520u -DLANES=64u netrun.comp -o netrun_big.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute xent.comp -o xent.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DDIM=128 xent_coop.comp -o xent_coop.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DDIM=64 xent_coop.comp -o xent_coop64.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DDIM=32 xent_coop.comp -o xent_coop32.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute tree.comp -o tree.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute xent_grad.comp -o xent_grad.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DDIM=32 xent_grad_coop.comp -o xent_grad_coop32.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DDIM=64 xent_grad_coop.comp -o xent_grad_coop64.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DDIM=128 xent_grad_coop.comp -o xent_grad_coop128.spv

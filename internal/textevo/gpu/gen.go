package gpu

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -I../../gpubrain -DMAX_NODES=2048u -DMAX_VALUES=2048u -DMAX_PLASTIC=1024u -DLANES=256u netrun.comp -o netrun.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute xent.comp -o xent.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DDIM=128 xent_coop.comp -o xent_coop.spv

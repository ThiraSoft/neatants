package gpu

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -I../../gpubrain -DMAX_VALUES=1024u -DMAX_MEMORY=256u -DMAX_PLASTIC=256u -DLEVEL_EDGES=256u -DLANES=64u netrun.comp -o netrun.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -I../../gpubrain -DMAX_VALUES=2048u -DMAX_MEMORY=1024u -DMAX_PLASTIC=1024u -DLEVEL_EDGES=256u -DLANES=64u netrun.comp -o netrun_big.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute xent.comp -o xent.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DDIM=128 xent_coop.comp -o xent_coop.spv

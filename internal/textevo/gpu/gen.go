package gpu

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -I../../gpubrain -DMAX_NODES=2048u -DMAX_VALUES=2048u -DMAX_PLASTIC=1024u netrun.comp -o netrun.spv

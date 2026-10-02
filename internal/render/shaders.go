package render

// Kage shaders. All use pixel units.

const terrainSrc = `//kage:unit pixels
package render

var Cam vec2
var Zoom float
var Time float
var Wet float
var Cave vec2
var Ponds [8]vec4
var Nests [6]vec4
var NestCol [6]vec4

func hash(p vec2) float {
	p3 := fract(vec3(p.x, p.y, p.x) * 0.1031)
	p3 += dot(p3, vec3(p3.y, p3.z, p3.x)+33.33)
	return fract((p3.x + p3.y) * p3.z)
}

func noise(p vec2) float {
	i := floor(p)
	f := fract(p)
	u := f * f * (3.0 - 2.0*f)
	a := hash(i)
	b := hash(i + vec2(1, 0))
	c := hash(i + vec2(0, 1))
	d := hash(i + vec2(1, 1))
	return mix(mix(a, b, u.x), mix(c, d, u.x), u.y)
}

func fbm(p vec2) float {
	v := 0.0
	a := 0.5
	for i := 0; i < 5; i++ {
		v += a * noise(p)
		p = p*2.03 + vec2(1.7, 9.2)
		a *= 0.5
	}
	return v
}

// voronoi returns (distance to nearest cell point, cell hash, second distance)
func voronoi(p vec2) vec3 {
	i := floor(p)
	f := fract(p)
	d1 := 8.0
	d2 := 8.0
	h := 0.0
	for y := -1; y <= 1; y++ {
		for x := -1; x <= 1; x++ {
			g := vec2(float(x), float(y))
			o := vec2(hash(i+g), hash(i+g+19.19))
			r := g + o - f
			d := length(r)
			if d < d1 {
				d2 = d1
				d1 = d
				h = hash(i + g + 7.7)
			} else if d < d2 {
				d2 = d
			}
		}
	}
	return vec3(d1, h, d2)
}

func Fragment(dstPos vec4, srcPos vec2, color vec4) vec4 {
	p := Cam + dstPos.xy/Zoom

	soil := vec3(0.40, 0.31, 0.21)
	dry := vec3(0.60, 0.53, 0.33)
	grass := vec3(0.30, 0.43, 0.15)
	lush := vec3(0.14, 0.29, 0.10)

	m := fbm(p * 0.0021)
	g := fbm(p*0.0055 + 10.0)
	col := mix(soil, dry, smoothstep(0.35, 0.68, m))

	// Wind waves over the grass
	wind := sin(dot(p, vec2(0.011, 0.006))-Time*1.2)*0.5 + 0.5
	wind2 := sin(dot(p, vec2(-0.004, 0.013))-Time*0.7)*0.5 + 0.5
	blade := noise(vec2(p.x*0.8+wind*2.0, p.y*0.16))
	blade2 := noise(vec2(p.x*1.9-wind2, p.y*0.35+3.0))
	grassCol := mix(lush, grass, blade*0.7+blade2*0.3)
	grassCol *= 0.8 + 0.35*wind*wind2 + 0.15*blade2
	grassAmt := smoothstep(0.42, 0.6, g)
	col = mix(col, grassCol, grassAmt)

	// Flowers in lush meadows
	fv := voronoi(p * 0.09)
	if grassAmt > 0.6 && fv.y > 0.93 && fv.x < 0.13 {
		fc := vec3(1.0, 0.85, 0.3)
		if fv.y > 0.965 {
			fc = vec3(0.95, 0.5, 0.75)
		} else if fv.y > 0.95 {
			fc = vec3(0.85, 0.9, 1.0)
		}
		col = mix(col, fc, smoothstep(0.13, 0.05, fv.x)*0.9)
	}

	detail := fbm(p * 0.07)
	col *= 0.82 + 0.34*detail
	col *= 0.93 + 0.14*noise(p*0.6)

	// Pebbles
	pv := voronoi(p * 0.06)
	if pv.y > 0.94 {
		stoneR := 0.1 + 0.16*(pv.y-0.94)*16.0
		st := smoothstep(stoneR, stoneR-0.05, pv.x)
		sc := vec3(0.47, 0.44, 0.4) * (0.7 + 0.4*hash(vec2(pv.y*91.0, 3.0)))
		sc *= 0.8 + 0.4*smoothstep(stoneR, 0.0, pv.x)
		col = mix(col*(1.0-0.18*smoothstep(stoneR+0.06, stoneR, pv.x)), sc, st)
	}

	// Cave: dark rock with glowing magma veins
	dc := length(p - Cave)
	rockN := fbm(p * 0.02)
	rock := smoothstep(300.0, 190.0, dc+rockN*90.0)
	if rock > 0.0 {
		rc := vec3(0.22, 0.2, 0.22) * (0.55 + 0.7*fbm(p*0.035))
		rv := voronoi(p * 0.018)
		rc *= 0.75 + 0.5*smoothstep(0.0, 0.4, rv.z-rv.x)
		vein := smoothstep(0.06, 0.0, rv.z-rv.x) * smoothstep(320.0, 80.0, dc)
		pulse := 0.6 + 0.4*sin(Time*2.0+dc*0.03)
		rc += vein * vec3(1.6, 0.45, 0.08) * pulse
		col = mix(col, rc, rock)
		hole := smoothstep(110.0, 55.0, dc+rockN*30.0)
		col = mix(col, vec3(0.02, 0.0, 0.01)+vec3(0.35, 0.05, 0.02)*pulse*smoothstep(0.0, 1.0, hole)*0.6, hole)
	}

	// Nest mounds
	for i := 0; i < 6; i++ {
		n := Nests[i]
		if n.z <= 0.0 {
			continue
		}
		d := length(p - n.xy)
		if d > n.z*3.0 {
			continue
		}
		nn := noise(p * 0.25)
		if n.w > 0.5 {
			mound := smoothstep(n.z*2.6, n.z*0.6, d+nn*8.0)
			mc := vec3(0.38, 0.27, 0.17) * (0.75 + 0.5*noise(p*0.5))
			mc *= 0.85 + 0.3*smoothstep(n.z*2.2, 0.0, d)
			col = mix(col, mc, mound*0.92)
			ring := smoothstep(6.0, 0.0, abs(d-n.z*1.15))
			col = mix(col, col*0.7+NestCol[i].rgb*0.2, ring*0.5)
			hole := smoothstep(n.z*0.36, n.z*0.2, d)
			col = mix(col, vec3(0.04, 0.025, 0.02), hole)
			for k := 0; k < 3; k++ {
				ang := float(k)*2.094 + n.x*0.01
				hp := n.xy + vec2(cos(ang), sin(ang))*n.z*0.75
				col = mix(col, vec3(0.06, 0.04, 0.03), smoothstep(5.0, 2.5, length(p-hp)))
			}
		} else {
			ash := smoothstep(n.z*2.2, 0.0, d+nn*20.0)
			col = mix(col, vec3(0.16, 0.14, 0.13)*(0.7+0.6*nn), ash*0.85)
		}
	}

	// Ponds
	for i := 0; i < 8; i++ {
		o := Ponds[i]
		if o.z <= 0.0 {
			continue
		}
		d := length(p-o.xy) - o.z + (noise(p*0.03)-0.5)*14.0
		if d > 24.0 {
			continue
		}
		if d < 0.0 {
			depth := clamp(-d/o.z*1.6, 0.0, 1.0)
			wc := mix(vec3(0.22, 0.45, 0.44), vec3(0.03, 0.12, 0.2), depth)
			q := p * 0.045
			cs := sin(q.x+Time*1.1+sin(q.y*0.8+Time*0.6)*2.0) * sin(q.y-Time*0.9+sin(q.x*0.7)*2.0)
			caust := pow(abs(cs), 6.0)
			wc += caust * vec3(0.25, 0.35, 0.35) * (1.0 - depth*0.6)
			sky := smoothstep(0.55, 0.9, fbm(p*0.004+vec2(Time*0.01, 0.0)))
			wc += sky * vec3(0.12, 0.14, 0.16)
			sp := noise(p*0.08 + vec2(Time*0.4, Time*0.2))
			wc += smoothstep(0.9, 1.0, sp) * 0.25
			// Rain ripples
			cell := floor(p / 34.0)
			loc := fract(p/34.0) - 0.5
			t := fract(Time*0.8 + hash(cell))
			rip := smoothstep(0.04, 0.0, abs(length(loc)-t*0.45)) * (1.0 - t)
			wc += rip * Wet * 0.5
			// Lily pads
			lv := voronoi(p * 0.02)
			if lv.y > 0.86 && depth < 0.8 {
				pad := smoothstep(0.24, 0.2, lv.x)
				wc = mix(wc, vec3(0.2, 0.42, 0.14)*(0.8+0.4*noise(p*0.3)), pad)
			}
			foam := smoothstep(4.0, 0.0, abs(d+3.0)) * (0.5 + 0.5*sin(Time*2.0+p.x*0.05))
			wc += foam * 0.25
			col = wc
		} else {
			col *= 0.62 + 0.38*smoothstep(0.0, 24.0, d)
		}
	}

	// Cloud shadows
	cloud := smoothstep(0.48, 0.78, fbm((p+Time*vec2(26.0, 11.0))*0.0011))
	col *= 1.0 - 0.3*cloud

	// Rain makes everything darker and glossier
	col *= 1.0 - 0.22*Wet
	col += Wet * 0.06 * smoothstep(0.7, 0.95, noise(p*0.08))

	// Outside world
	if p.x < 0.0 || p.y < 0.0 || p.x > 3600.0 || p.y > 2200.0 {
		col *= 0.25
	}
	return vec4(col, 1)
}
`

const blurSrc = `//kage:unit pixels
package render

var Dir vec2

func Fragment(dstPos vec4, srcPos vec2, color vec4) vec4 {
	o := imageSrc0Origin()
	s := imageSrc0Size()
	l := srcPos - o
	c := vec4(0)
	w := 0.0
	for i := -6; i <= 6; i++ {
		f := float(i)
		k := exp(-f * f / 18.0)
		q := clamp(l+Dir*f, vec2(0.5), s-0.5)
		c += imageSrc0At(q+o) * k
		w += k
	}
	return c / w
}
`

const compositeSrc = `//kage:unit pixels
package render

var Ambient vec3
var LightK float
var Time float
var Shocks [16]vec4
var Chroma float
var Flash float
var Night float
var Bloom float

func hash(p vec2) float {
	p3 := fract(vec3(p.x, p.y, p.x) * 0.1031)
	p3 += dot(p3, vec3(p3.y, p3.z, p3.x)+33.33)
	return fract((p3.x + p3.y) * p3.z)
}

func at0(l vec2) vec4 {
	return imageSrc0At(clamp(l, vec2(0.5), imageSrc0Size()-0.5) + imageSrc0Origin())
}
func at1(l vec2) vec4 {
	return imageSrc1At(clamp(l, vec2(0.5), imageSrc1Size()-0.5) + imageSrc1Origin())
}
func at2(l vec2) vec4 {
	return imageSrc2At(clamp(l, vec2(0.5), imageSrc2Size()-0.5) + imageSrc2Origin())
}
func at3(l vec2) vec4 {
	return imageSrc3At(clamp(l, vec2(0.5), imageSrc3Size()-0.5) + imageSrc3Origin())
}

func Fragment(dstPos vec4, srcPos vec2, color vec4) vec4 {
	size := imageSrc0Size()
	uv := srcPos - imageSrc0Origin()

	off := vec2(0)
	glow := 0.0
	for i := 0; i < 16; i++ {
		s := Shocks[i]
		if s.w != 0.0 {
			d := uv - s.xy
			l := length(d) + 0.001
			x := (l - s.z) / 22.0
			ring := exp(-x * x)
			off += d / l * ring * s.w
			glow += ring * abs(s.w) * 0.012
		}
	}
	p := uv + off
	center := size * 0.5
	cd := (uv - center) / size.y
	ca := (Chroma + length(off)*0.08) * (0.3 + length(cd))
	dir := normalize(cd+0.0001) * ca

	scene := vec3(at0(p+dir).r, at0(p).g, at0(p-dir).b)
	light := at1(p).rgb
	fx := vec3(at2(p+dir*1.5).r, at2(p).g, at2(p-dir*1.5).b)
	bloom := at3(uv).rgb

	lit := scene * (Ambient + light*LightK)
	c := lit + fx + bloom*Bloom + vec3(Flash*0.9, Flash*0.95, Flash) + glow

	// Filmic tonemap (ACES fit)
	c *= 1.08
	c = clamp((c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14), 0.0, 1.0)

	// Grading: warm highlights by day, teal shadows by night
	lum := dot(c, vec3(0.299, 0.587, 0.114))
	c = mix(c, c*vec3(1.04, 1.0, 0.94), (1.0-Night)*0.6)
	c = mix(c, mix(vec3(0.02, 0.05, 0.1), c, smoothstep(0.0, 0.45, lum)+0.35), Night*0.5)

	// Vignette and grain
	v := length(cd * vec2(0.9, 1.1))
	c *= 1.0 - 0.55*pow(v, 2.4)
	c += (hash(uv+fract(Time)*97.0) - 0.5) * 0.028
	return vec4(c, 1)
}
`

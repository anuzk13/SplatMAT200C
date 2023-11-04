let cameras = [
	{
		id: 0,
		img_name: "00001",
		width: 1959,
		height: 1090,
		position: [
			-3.0089893469241797, -0.11086489695181866, -3.7527640949141428,
		],
		rotation: [
			[0.876134201218856, 0.06925962026449776, 0.47706599800804744],
			[-0.04747421839895102, 0.9972110940209488, -0.057586739349882114],
			[-0.4797239414934443, 0.027805376500959853, 0.8769787916452908],
		],
		fy: 1164.6601287484507,
		fx: 1159.5880733038064,
	}
];

function DynamicCamera(orbitEl, zIsUp, {
  phiRad = 0, thetaRad = 0, distance = 2, target = [0, 0, 0]
} = {}) {
  // parameters not used in practice
  const camera = new THREE.PerspectiveCamera(45, 640 / 480, 1, 10000);
  camera.up = new THREE.Vector3(0, 1, 0);
  camera.position.y = 1;

  const orbitControls = new THREE.OrbitControls(camera, orbitEl);
  orbitControls.update();

  const sp = new THREE.Spherical();
  sp.radius = orbitControls.getDistance();
  sp.phi = orbitControls.getPolarAngle();
  sp.theta = orbitControls.getAzimuthalAngle();

  sp.phi = Math.PI / 2 - phiRad;
  sp.theta = thetaRad - Math.PI / 2;
  sp.radius = distance;
  orbitControls.object.position.setFromSpherical(sp);
  orbitControls.update();

  let changed = true;

  orbitControls.addEventListener('change', () => {
    changed = true;
  });

  this.wasChanged = () => {
    const was = changed;
    changed = false;
    return was;
  };
  
  function getTargetCoordinatesActual() {
    return [
      orbitControls.target.x + target[0],
      -orbitControls.target.z + target[1],
      orbitControls.target.y + target[2]
    ];
  }
  
  this.getCameraTarget = () => {
    const t = getTargetCoordinatesActual();
    if (zIsUp) {
      return { x: t[0], y: t[1], z: t[2] };
    } else {
      // ugly
      return { x: t[0], y: -t[2], z: t[1] };
    }
  };

  this.getViewMatrix = () => {
    const a1 = -orbitControls.getAzimuthalAngle();
    const a2 = Math.PI / 2 - orbitControls.getPolarAngle();
    const d = orbitControls.getDistance();

    const rotAzimuth = [[Math.cos(a1), Math.sin(a1)], [-Math.sin(a1), Math.cos(a1)]];

    function applyRotAzimuth(v) {
      const r = rotAzimuth;
      return [r[0][0] * v[0] + r[0][1] * v[1], r[1][0] * v[0] + r[1][1] * v[1], v[2]];
    };

    const dirFwd = [0, Math.cos(a2), -Math.sin(a2)];
    const dirRight = [1, 0, 0];
    const dirDown = [0, -Math.sin(a2), -Math.cos(a2)];
    const trg = getTargetCoordinatesActual();

    const camZ = applyRotAzimuth(dirFwd);
    const pos = [-camZ[0] * d + trg[0], -camZ[1] * d + trg[1], -camZ[2] * d + trg[2]];

    const out = {
      pos: pos,
      x: applyRotAzimuth(dirRight),
      y: applyRotAzimuth(dirDown),
      z: camZ
    };

		// console.log({'pos': orbitControls.target});

		function transpose4(m) {
			const m1 = translate4(m, 0, 0, 0); // copy
			for (let i = 0; i < 4; ++i)
				for (let j = 0; j < 4; ++j) {
					m1[4*j+i] = m[4*i+j];
				}
		  return m1;
		}

		const Y_IS_UP_TO_Z_IS_UP = transpose4([
			1, 0, 0, 0,
			0, 0,-1, 0,
			0, 1, 0, 0,
			0, 0, 0, 1
		]);

		const camToWorld = transpose4([
			out.x[0], out.y[0], out.z[0], out.pos[0],
			out.x[1], out.y[1], out.z[1], out.pos[1],
			out.x[2], out.y[2], out.z[2], out.pos[2],
			0, 0, 0, 1
		]);

		if (!zIsUp)
			return invert4(multiply4(Y_IS_UP_TO_Z_IS_UP, camToWorld));

		return invert4(camToWorld);
  };
}
function createDatGui(onBallUpdate, cameraControls) {

  // Function to be controlled
  const settings = {
      bgColor: '#000000',
		  ballX: 0.0,
		  ballY: 0.0,
		  ballZ: 0.0,
		  ballRadius: 0
  };


	// Function to change the background color
	function changeBackgroundColor(color) {
	    document.body.style.backgroundColor = color;
	}

	// Create a new dat.GUI instance
	const gui = new dat.GUI();

	// Add a color controller
	const colorController = gui.addColor(settings, 'bgColor');

	// Listen to changes on the color controller
	colorController.onChange(function(value) {
	    changeBackgroundColor(value);
	});

	const RANGE = 10;
	['X', 'Y', 'Z'].forEach(c => {
			gui.add(settings, `ball${c}`, -RANGE, RANGE).step(RANGE/1000).onChange(() => onBallUpdate(settings));
	});
	
	gui.add(settings, 'ballRadius', 0, RANGE).step(RANGE/500).onChange(() => onBallUpdate(settings));
	
	
	function setBallCenterToCameraTarget() {
	  const center = cameraControls.getCameraTarget();
    settings.ballX = center.x;
    settings.ballY = center.y;
    settings.ballZ = center.z;
    // Update GUI display for each controller
    for (let i in gui.__controllers) {
      gui.__controllers[i].updateDisplay();
    }
    onBallUpdate(settings);
  }
	
	gui.add({ resetCertain: setBallCenterToCameraTarget }, 'resetCertain').name('Set to camera target');

	changeBackgroundColor(settings.color);
}

const camera = cameras[0];

function getProjectionMatrix(fx, fy, width, height) {
	const znear = 0.2;
	const zfar = 200;
	return [
		[(2 * fx) / width, 0, 0, 0],
		[0, -(2 * fy) / height, 0, 0],
		[0, 0, zfar / (zfar - znear), 1],
		[0, 0, -(zfar * znear) / (zfar - znear), 0],
	].flat();
}

function getViewMatrix(camera) {
	const R = camera.rotation.flat();
	const t = camera.position;
	const camToWorld = [
		[R[0], R[1], R[2], 0],
		[R[3], R[4], R[5], 0],
		[R[6], R[7], R[8], 0],
		[
			-t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
			-t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
			-t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
			1,
		],
	].flat();
	return camToWorld;
}

function multiply4(a, b) {
	return [
		b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
		b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
		b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
		b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
		b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
		b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
		b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
		b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
		b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
		b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
		b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
		b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
		b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
		b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
		b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
		b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
	];
}

function invert4(a) {
	let b00 = a[0] * a[5] - a[1] * a[4];
	let b01 = a[0] * a[6] - a[2] * a[4];
	let b02 = a[0] * a[7] - a[3] * a[4];
	let b03 = a[1] * a[6] - a[2] * a[5];
	let b04 = a[1] * a[7] - a[3] * a[5];
	let b05 = a[2] * a[7] - a[3] * a[6];
	let b06 = a[8] * a[13] - a[9] * a[12];
	let b07 = a[8] * a[14] - a[10] * a[12];
	let b08 = a[8] * a[15] - a[11] * a[12];
	let b09 = a[9] * a[14] - a[10] * a[13];
	let b10 = a[9] * a[15] - a[11] * a[13];
	let b11 = a[10] * a[15] - a[11] * a[14];
	let det =
		b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
	if (!det) return null;
	return [
		(a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
		(a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
		(a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
		(a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
		(a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
		(a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
		(a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
		(a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
		(a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
		(a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
		(a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
		(a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
		(a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
		(a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
		(a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
		(a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
	];
}

function rotate4(a, rad, x, y, z) {
	let len = Math.hypot(x, y, z);
	x /= len;
	y /= len;
	z /= len;
	let s = Math.sin(rad);
	let c = Math.cos(rad);
	let t = 1 - c;
	let b00 = x * x * t + c;
	let b01 = y * x * t + z * s;
	let b02 = z * x * t - y * s;
	let b10 = x * y * t - z * s;
	let b11 = y * y * t + c;
	let b12 = z * y * t + x * s;
	let b20 = x * z * t + y * s;
	let b21 = y * z * t - x * s;
	let b22 = z * z * t + c;
	return [
		a[0] * b00 + a[4] * b01 + a[8] * b02,
		a[1] * b00 + a[5] * b01 + a[9] * b02,
		a[2] * b00 + a[6] * b01 + a[10] * b02,
		a[3] * b00 + a[7] * b01 + a[11] * b02,
		a[0] * b10 + a[4] * b11 + a[8] * b12,
		a[1] * b10 + a[5] * b11 + a[9] * b12,
		a[2] * b10 + a[6] * b11 + a[10] * b12,
		a[3] * b10 + a[7] * b11 + a[11] * b12,
		a[0] * b20 + a[4] * b21 + a[8] * b22,
		a[1] * b20 + a[5] * b21 + a[9] * b22,
		a[2] * b20 + a[6] * b21 + a[10] * b22,
		a[3] * b20 + a[7] * b21 + a[11] * b22,
		...a.slice(12, 16),
	];
}

function translate4(a, x, y, z) {
	return [
		...a.slice(0, 12),
		a[0] * x + a[4] * y + a[8] * z + a[12],
		a[1] * x + a[5] * y + a[9] * z + a[13],
		a[2] * x + a[6] * y + a[10] * z + a[14],
		a[3] * x + a[7] * y + a[11] * z + a[15],
	];
}

function getShaderGlsl(nSphericalHarmonics = 0) {
		let shAttributesGlsl = "";
		let getColorGlsl = "";
		let callGetColorGlsl = "color";

		const shFormula = [
			"0.28209479177387814",
			"-0.48860251190291987 * ray.y",
			"0.48860251190291987 * ray.z",
			"-0.48860251190291987 * ray.x",
			"1.0925484305920792 * ray.x * ray.y",
			"-1.0925484305920792 * ray.y * ray.z",
			"0.94617469575755997 * ray.z * ray.z - 0.31539156525251999",
			"-1.0925484305920792 * ray.x * ray.z",
			"0.54627421529603959 * ray.x * ray.x - 0.54627421529603959 * ray.y * ray.y",
			"0.59004358992664352 * ray.y * (-3.0 * ray.x * ray.x + ray.y * ray.y)",
			"2.8906114426405538 * ray.x * ray.y * ray.z",
			"0.45704579946446572 * ray.y * (1.0 - 5.0 * ray.z * ray.z)",
			"0.3731763325901154 * ray.z * (5.0 * ray.z * ray.z - 3.0)",
			"0.45704579946446572 * ray.x * (1.0 - 5.0 * ray.z * ray.z)",
			"1.4453057213202769 * ray.z * (ray.x * ray.x - ray.y * ray.y)",
			"0.59004358992664352 * ray.x * (-ray.x * ray.x + 3.0 * ray.y * ray.y)",
		];

		if (nSphericalHarmonics > 0) {
			for (let i = 0; i < nSphericalHarmonics; ++i) {
				shAttributesGlsl += `attribute vec3 sh${i};
				`;
			}

			// const colorPost = "";
			const colorPost = "activation = activation * 5.0;";
			// const colorPost = "activation = activation - 0.5;";

			let tail = '';
			for (let i = 1; i < nSphericalHarmonics; ++i)
				tail += `activation += (${shFormula[i]}) * sh${i};
				`;

			getColorGlsl = `
			vec3 getColor(vec3 ray) {
					vec3 activation = (${shFormula[0]}) * sh0;
					${tail}
					${colorPost}
					return 1.0 / (1.0 + exp(-activation));
			}
			`;

			callGetColorGlsl = `vec4(getColor(vec3(view[0][2], view[1][2], view[2][2])), color.a)`;
		}

		const vertexShaderSource = `
		precision mediump float;
		attribute vec2 position;

		attribute vec4 color;
		attribute vec3 center;
		attribute vec3 covA;
		attribute vec3 covB;
		${shAttributesGlsl}

		uniform mat4 projection, view;
		uniform vec2 focal;
		uniform vec2 viewport;

		varying vec4 vColor;
		varying vec2 vPosition;

		${getColorGlsl}

		mat3 transpose(mat3 m) {
				return mat3(
				    m[0][0], m[1][0], m[2][0],
				    m[0][1], m[1][1], m[2][1],
				    m[0][2], m[1][2], m[2][2]
				);
		}

		void main () {
				vec4 camspace = view * vec4(center, 1);
				vec4 pos2d = projection * camspace;

				float bounds = 1.2 * pos2d.w;
				if (pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds
					 || pos2d.y < -bounds || pos2d.y > bounds) {
					    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
					    return;
				}

				mat3 Vrk = mat3(
				    covA.x, covA.y, covA.z,
				    covA.y, covB.x, covB.y,
				    covA.z, covB.y, covB.z
				);

				mat3 J = mat3(
				    focal.x / camspace.z, 0., -(focal.x * camspace.x) / (camspace.z * camspace.z),
				    0., -focal.y / camspace.z, (focal.y * camspace.y) / (camspace.z * camspace.z),
				    0., 0., 0.
				);

				mat3 W = transpose(mat3(view));
				mat3 T = W * J;
				mat3 cov = transpose(T) * Vrk * T;

				vec2 vCenter = vec2(pos2d) / pos2d.w;

				float diagonal1 = cov[0][0] + 0.3;
				float offDiagonal = cov[0][1];
				float diagonal2 = cov[1][1] + 0.3;

				float mid = 0.5 * (diagonal1 + diagonal2);
				float radius = length(vec2((diagonal1 - diagonal2) / 2.0, offDiagonal));
				float lambda1 = mid + radius;
				float lambda2 = max(mid - radius, 0.1);
				vec2 diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1));
				vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
				vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);


				vColor = ${callGetColorGlsl};
				vPosition = position;

				gl_Position = vec4(
				    vCenter
				        + position.x * v1 / viewport * 2.0
				        + position.y * v2 / viewport * 2.0, 0.0, 1.0);
		}
		`;

		const fragmentShaderSource = `
		precision mediump float;

		varying vec4 vColor;
		varying vec2 vPosition;

		void main () {
				float A = -dot(vPosition, vPosition);
				if (A < -4.0) discard;
				float B = exp(A) * vColor.a;
				// gl_FragColor = vec4(B * vec3(vPosition, 1.0), B);
				gl_FragColor = vec4(B * vColor.rgb, B);
		}
		`;

		// console.log(vertexShaderSource);
		return { frag: fragmentShaderSource, vert: vertexShaderSource };
}

function createWorker(self) {
	let buffer;
	let vertexCount = 0;
	let viewProj;
	// 6*4 + 4 + 4 = 8*4
	// XYZ - Position (Float32)
	// XYZ - Scale (Float32)
	// RGBA - colors (uint8)
	// IJKL - quaternion/rot (uint8)
	let lastProj = [];
	let depthIndex = new Uint32Array();
	let sphericalHarmonics;
	let settings = {};

	function getFloatRowLength() {
	    return 3 + 3 + (sphericalHarmonics ? (3 * sphericalHarmonics.nPerChannel) : 0);
	}

	function getRowLength() {
      return getFloatRowLength() * 4 + 4 + 4;
	}

	function acceptSplat({ x, y, z }) {
	    const r = settings.ballRadius;
	    if (!r) return true;
	    const dx = x - settings.ballX;
	    const dy = y - settings.ballY;
	    const dz = z - settings.ballZ;
			return dx*dx + dy*dy + dz*dz < r*r;
	}

	function runFilter(floatBuffer, stride, length) {
			let n = 0;
			const mask = Array.from({ length }, (_, i) => {
		    const x = floatBuffer[stride * i + 0];
		    const y = floatBuffer[stride * i + 1];
		    const z = floatBuffer[stride * i + 2];

		    if (acceptSplat({ x, y, z })) {
			    n++;
			    return true;
		    }
		    return false;
	    });
	    return { n, mask };
	}

	const runSort = (viewProj) => {
		if (!buffer) return;

		const fBufStride = getFloatRowLength() + 2;
		const uBufStride = fBufStride * 4;
		const uBufOffset = getFloatRowLength() * 4;

		// console.log({sphericalHarmonics,fBufStride,uBufStride,uBufOffset});

		const f_buffer = new Float32Array(buffer);
		const u_buffer = new Uint8Array(buffer);

		const { n, mask } = runFilter(f_buffer, fBufStride, vertexCount);

		const covA = new Float32Array(3 * n);
		const covB = new Float32Array(3 * n);

		const center = new Float32Array(3 * n);
		const color = new Float32Array(4 * n);

		let shBuffers;

		if (sphericalHarmonics) {
		    shBuffers = Array.from({ length: sphericalHarmonics.nPerChannel }, (_, i) => new Float32Array(3 * n));
		}

		if (depthIndex.length == n) {
			let dot =
				lastProj[2] * viewProj[2] +
				lastProj[6] * viewProj[6] +
				lastProj[10] * viewProj[10];
			if (Math.abs(dot - 1) < 0.01) {
				return;
			}
		}

		let maxDepth = -Infinity;
		let minDepth = Infinity;
		let sizeList = new Int32Array(vertexCount);
		for (let i = 0; i < vertexCount; i++) {
			let depth =
				((viewProj[2] * f_buffer[fBufStride * i + 0] +
					viewProj[6] * f_buffer[fBufStride * i + 1] +
					viewProj[10] * f_buffer[fBufStride * i + 2]) *
					4096) |
				0;
			sizeList[i] = depth;
			if (depth > maxDepth) maxDepth = depth;
			if (depth < minDepth) minDepth = depth;
		}
		// console.time("sort");

		// This is a 16 bit single-pass counting sort
		let depthInv = (256 * 256) / (maxDepth - minDepth);
		let counts0 = new Uint32Array(256*256);
		for (let i = 0; i < vertexCount; i++) {
			if (mask[i]) {
					sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
					counts0[sizeList[i]]++;
			}
		}
		let starts0 = new Uint32Array(256*256);
		for (let i = 1; i < 256*256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];
		depthIndex = new Uint32Array(n);
		for (let i = 0; i < vertexCount; i++) {
				if (mask[i])
						depthIndex[starts0[sizeList[i]]++] = i;
		}


		lastProj = viewProj;
		// console.timeEnd("sort");
		for (let j = 0; j < n; j++) {
			const i = depthIndex[j];

			center[3 * j + 0] = f_buffer[fBufStride * i + 0];
			center[3 * j + 1] = f_buffer[fBufStride * i + 1];
			center[3 * j + 2] = f_buffer[fBufStride * i + 2];

			color[4 * j + 0] = u_buffer[uBufStride * i + uBufOffset + 0] / 255;
			color[4 * j + 1] = u_buffer[uBufStride * i + uBufOffset + 1] / 255;
			color[4 * j + 2] = u_buffer[uBufStride * i + uBufOffset + 2] / 255;
			color[4 * j + 3] = u_buffer[uBufStride * i + uBufOffset + 3] / 255;

			let scale = [
				f_buffer[fBufStride * i + 3 + 0],
				f_buffer[fBufStride * i + 3 + 1],
				f_buffer[fBufStride * i + 3 + 2],
			];
			let rot = [
				(u_buffer[uBufStride * i + uBufOffset + 4 + 0] - 128) / 128,
				(u_buffer[uBufStride * i + uBufOffset + 4 + 1] - 128) / 128,
				(u_buffer[uBufStride * i + uBufOffset + 4 + 2] - 128) / 128,
				(u_buffer[uBufStride * i + uBufOffset + 4 + 3] - 128) / 128,
			];

			if (shBuffers) {
			    for (let jj = 0; jj < shBuffers.length; ++jj) {
			        for (let kk = 0; kk < 3; ++kk) {
			            shBuffers[jj][3 * j + kk] = f_buffer[fBufStride * i + 6 + jj*3 + kk];
			        }
			    }
			}

			const R = [
				1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
				2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
				2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

				2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
				1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
				2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

				2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
				2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
				1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
			];

			const MAX_SCALE = 10000;
			// Compute the matrix product of S and R (M = S * R)
			if (scale[0]*scale[0] + scale[1]*scale[1] + scale[2]*scale[2] > MAX_SCALE*MAX_SCALE) {
			    throw "Invalid scale, the data is probably corrupted";
			}

			/*const CAP_SCALE = 0.001;
			for (let iii = 0; iii < 3; ++iii)
			    scale[iii] = Math.min(scale[iii], CAP_SCALE);*/

			const M = [
				scale[0] * R[0],
				scale[0] * R[1],
				scale[0] * R[2],
				scale[1] * R[3],
				scale[1] * R[4],
				scale[1] * R[5],
				scale[2] * R[6],
				scale[2] * R[7],
				scale[2] * R[8],
			];

			covA[3 * j + 0] = M[0] * M[0] + M[3] * M[3] + M[6] * M[6];
			covA[3 * j + 1] = M[0] * M[1] + M[3] * M[4] + M[6] * M[7];
			covA[3 * j + 2] = M[0] * M[2] + M[3] * M[5] + M[6] * M[8];
			covB[3 * j + 0] = M[1] * M[1] + M[4] * M[4] + M[7] * M[7];
			covB[3 * j + 1] = M[1] * M[2] + M[4] * M[5] + M[7] * M[8];
			covB[3 * j + 2] = M[2] * M[2] + M[5] * M[5] + M[8] * M[8];
		}

		let objs = { covA, center, color, covB, viewProj };
		let bufs = [
			covA.buffer,
			center.buffer,
			color.buffer,
			covB.buffer
		];
		if (shBuffers) {
		    for (let jj = 0; jj < shBuffers.length; ++jj) {
		        objs[`sh${jj}`] = shBuffers[jj];
		        bufs.push(shBuffers[jj].buffer);
		    }
		}

		self.postMessage(objs, bufs);

		// console.timeEnd("sort");
	};

	function processPlyBuffer(inputBuffer) {
        const floatRowLength = getFloatRowLength();
        const rowLength = getRowLength();

		const ubuf = new Uint8Array(inputBuffer);
		// 10KB ought to be enough for a header...
		const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
		const header_end = "end_header\n";
		const header_end_index = header.indexOf(header_end);
		if (header_end_index < 0)
			throw new Error("Unable to read .ply file header");
		const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
		console.log("Vertex Count", vertexCount);
		let row_offset = 0,
			offsets = {},
			types = {};
		const TYPE_MAP = {
			double: "getFloat64",
			int: "getInt32",
			uint: "getUint32",
			float: "getFloat32",
			short: "getInt16",
			ushort: "getUint16",
			uchar: "getUint8",
		};
		for (let prop of header
			.slice(0, header_end_index)
			.split("\n")
			.filter((k) => k.startsWith("property "))) {
			const [p, type, name] = prop.split(" ");
			const arrayType = TYPE_MAP[type] || "getInt8";
			types[name] = arrayType;
			offsets[name] = row_offset;
			row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
		}
		console.log("Bytes per row", row_offset, types, offsets);

		let dataView = new DataView(
			inputBuffer,
			header_end_index + header_end.length,
		);
		let row = 0;
		const attrs = new Proxy(
			{},
			{
				get(target, prop) {
					if (!types[prop]) throw new Error(prop + " not found");
					return dataView[types[prop]](
						row * row_offset + offsets[prop],
						true,
					);
				},
			},
		);

		console.time("calculate importance");
		let sizeList = new Float32Array(vertexCount);
		let sizeIndex = new Uint32Array(vertexCount);
		for (row = 0; row < vertexCount; row++) {
			sizeIndex[row] = row;
			if (!types["scale_0"]) continue;
			const size =
				Math.exp(attrs.scale_0) *
				Math.exp(attrs.scale_1) *
				Math.exp(attrs.scale_2);
			const opacity = 1 / (1 + Math.exp(-attrs.opacity));
			sizeList[row] = size * opacity;
		}
		console.timeEnd("calculate importance");

		console.time("sort");
		sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
		console.timeEnd("sort");

		// 6*4 + 4 + 4 = 8*4
		// XYZ - Position (Float32)
		// XYZ - Scale (Float32)
		// RGBA - colors (uint8)
		// IJKL - quaternion/rot (uint8)
		const buffer = new ArrayBuffer(rowLength * vertexCount);

		console.time("build buffer");
		for (let j = 0; j < vertexCount; j++) {
			row = sizeIndex[j];

			const position = new Float32Array(buffer, j * rowLength, 3);
			const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
			const rgba = new Uint8ClampedArray(
				buffer,
				j * rowLength + floatRowLength,
				4,
			);
			const rot = new Uint8ClampedArray(
				buffer,
				j * rowLength + floatRowLength + 4,
				4,
			);

			if (types["scale_0"]) {
				const qlen = Math.sqrt(
					attrs.rot_0 ** 2 +
						attrs.rot_1 ** 2 +
						attrs.rot_2 ** 2 +
						attrs.rot_3 ** 2,
				);

				rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
				rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
				rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
				rot[3] = (attrs.rot_3 / qlen) * 128 + 128;

				scales[0] = Math.exp(attrs.scale_0);
				scales[1] = Math.exp(attrs.scale_1);
				scales[2] = Math.exp(attrs.scale_2);
			} else {
				scales[0] = 0.01;
				scales[1] = 0.01;
				scales[2] = 0.01;

				rot[0] = 255;
				rot[1] = 0;
				rot[2] = 0;
				rot[3] = 0;
			}

			position[0] = attrs.x;
			position[1] = attrs.y;
			position[2] = attrs.z;

			if (types["f_dc_0"]) {
				const SH_C0 = 0.28209479177387814;
				rgba[0] = (0.5 + SH_C0 * attrs.f_dc_0) * 255;
				rgba[1] = (0.5 + SH_C0 * attrs.f_dc_1) * 255;
				rgba[2] = (0.5 + SH_C0 * attrs.f_dc_2) * 255;
			} else {
				rgba[0] = attrs.red;
				rgba[1] = attrs.green;
				rgba[2] = attrs.blue;
			}
			if (types["opacity"]) {
				rgba[3] = (1 / (1 + Math.exp(-attrs.opacity))) * 255;
			} else {
				rgba[3] = 255;
			}
		}
		console.timeEnd("build buffer");
		return buffer;
	}

	const throttledSort = () => {
		if (!sortRunning) {
			sortRunning = true;
			let lastView = viewProj;
			runSort(lastView);
			setTimeout(() => {
				sortRunning = false;
				if (lastView !== viewProj) {
					throttledSort();
				}
			}, 0);
		}
	};

	let sortRunning;
	self.onmessage = (e) => {
	  if (e.data.settings) {
	    settings = e.data.settings;
	    console.log({settings});
	    throttledSort();
	  } else if (e.data.ply) {
			vertexCount = 0;
			runSort(viewProj);
			buffer = processPlyBuffer(e.data.ply);
			vertexCount = Math.floor(buffer.byteLength / getRowLength());
			postMessage({ buffer: buffer });
		} else if (e.data.buffer) {
			buffer = e.data.buffer;
			vertexCount = e.data.vertexCount;
			sphericalHarmonics = e.data.sphericalHarmonics.keep ? e.data.sphericalHarmonics : null;
		} else if (e.data.vertexCount) {
			vertexCount = e.data.vertexCount;
		} else if (e.data.view) {
			viewProj = e.data.view;
			throttledSort();
		}
	};
}

let defaultViewMatrix = [
	0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07,
	0.03, 6.55, 1,
];

let activeDownsample = null
async function main() {
	let carousel = true;
	const params = new URLSearchParams(location.search);
	try {
		viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
		carousel = false;
	} catch (err) {}
	const url = new URL(
		// "nike.splat",
		// location.href,
		params.get("url") || "train.splat",
		"https://huggingface.co/cakewalk/splat-data/resolve/main/",
	);
	const req = await fetch(url, {
		mode: "cors", // no-cors, *cors, same-origin
		credentials: "omit", // include, *same-origin, omit
	});
	console.log(req);
	if (req.status != 200)
		throw new Error(req.status + " Unable to load " + req.url);

	const reader = req.body.getReader();
	let splatData = new Uint8Array(req.headers.get("content-length"));

	const N_HARMONICS_PER_CHANNEL_IN_FILE = 16;
	const sphericalHarmonics = {
		keep: !!params.get('sh'),
		nPerChannel: N_HARMONICS_PER_CHANNEL_IN_FILE,
		nRendered: 9
	};
	const rowLength = (3 + 3 + (sphericalHarmonics.keep ? (3 * sphericalHarmonics.nPerChannel) : 0)) * 4 + 4 + 4;

	const downsample =
		splatData.length / rowLength > 500000 ? 1 : 1 / devicePixelRatio;
	// const downsample = 1 / devicePixelRatio;
	// const downsample = 1;
	console.log(splatData.length / rowLength, downsample);

	const worker = new Worker(
		URL.createObjectURL(
			new Blob(["(", createWorker.toString(), ")(self)"], {
				type: "application/javascript",
			}),
		),
	);

	const canvas = document.getElementById("canvas");
	canvas.width = innerWidth / downsample;
	canvas.height = innerHeight / downsample;

	let dyncam = new DynamicCamera(canvas, !!params.get("z_is_up"));
	let viewMatrix = dyncam.getViewMatrix();

	const fps = document.getElementById("fps");

	let projectionMatrix = getProjectionMatrix(
		camera.fx / downsample,
		camera.fy / downsample,
		canvas.width,
		canvas.height,
	);

	const gl = canvas.getContext("webgl");
	const ext = gl.getExtension("ANGLE_instanced_arrays");

	const shaderSource = getShaderGlsl(sphericalHarmonics.keep ? sphericalHarmonics.nRendered : 0);

	const vertexShader = gl.createShader(gl.VERTEX_SHADER);
	gl.shaderSource(vertexShader, shaderSource.vert);
	gl.compileShader(vertexShader);
	if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
		console.error(gl.getShaderInfoLog(vertexShader));

	const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
	gl.shaderSource(fragmentShader, shaderSource.frag);
	gl.compileShader(fragmentShader);
	if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
		console.error(gl.getShaderInfoLog(fragmentShader));

	const program = gl.createProgram();
	gl.attachShader(program, vertexShader);
	gl.attachShader(program, fragmentShader);
	gl.linkProgram(program);
	gl.useProgram(program);

	if (!gl.getProgramParameter(program, gl.LINK_STATUS))
		console.error(gl.getProgramInfoLog(program));

	gl.disable(gl.DEPTH_TEST); // Disable depth testing

	// Enable blending
	gl.enable(gl.BLEND);

	// Set blending function
	gl.blendFuncSeparate(
		gl.ONE_MINUS_DST_ALPHA,
		gl.ONE,
		gl.ONE_MINUS_DST_ALPHA,
		gl.ONE,
	);

	// Set blending equation
	gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

	// projection
	const u_projection = gl.getUniformLocation(program, "projection");
	gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

	// viewport
	const u_viewport = gl.getUniformLocation(program, "viewport");
	gl.uniform2fv(u_viewport, new Float32Array([canvas.width, canvas.height]));

	// focal
	const u_focal = gl.getUniformLocation(program, "focal");
	gl.uniform2fv(
		u_focal,
		new Float32Array([camera.fx / downsample, camera.fy / downsample]),
	);

	// view
	const u_view = gl.getUniformLocation(program, "view");
	gl.uniformMatrix4fv(u_view, false, viewMatrix);

	// positions
	const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
	const vertexBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
	gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
	const a_position = gl.getAttribLocation(program, "position");
	gl.enableVertexAttribArray(a_position);
	gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
	gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);


	function createVecFloatBuffer(name, vecDim) {
			const buf = gl.createBuffer();
			const a = gl.getAttribLocation(program, name);
			gl.enableVertexAttribArray(a);
			gl.bindBuffer(gl.ARRAY_BUFFER, buf);
			gl.vertexAttribPointer(a, vecDim, gl.FLOAT, false, 0, 0);
			ext.vertexAttribDivisorANGLE(a, 1);
			return buf;
	}


	const centerBuffer = createVecFloatBuffer('center', 3);
	const colorBuffer = createVecFloatBuffer('color', 4);
	const covABuffer = createVecFloatBuffer('covA', 3);
	const covBBuffer = createVecFloatBuffer('covB', 3);
	const shBuffersGl = [];
	for (let i = 0; i < sphericalHarmonics.nRendered; ++i)
			shBuffersGl.push(createVecFloatBuffer(`sh${i}`, 3));

	let lastProj = [];
	let lastData;

	worker.onmessage = (e) => {
		if (e.data.buffer) {
			splatData = new Uint8Array(e.data.buffer);
			const blob = new Blob([splatData.buffer], {
				type: "application/octet-stream",
			});
			const link = document.createElement("a");
			link.download = "model.splat";
			link.href = URL.createObjectURL(blob);
			document.body.appendChild(link);
			link.click();
		} else {
			let { covA, covB, center, color, viewProj, ...shBufferData } = e.data;
			lastData = e.data;

			activeDownsample = downsample

			lastProj = viewProj;
			vertexCount = center.length / 3;

			gl.bindBuffer(gl.ARRAY_BUFFER, centerBuffer);
			gl.bufferData(gl.ARRAY_BUFFER, center, gl.DYNAMIC_DRAW);

			gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
			gl.bufferData(gl.ARRAY_BUFFER, color, gl.DYNAMIC_DRAW);

			gl.bindBuffer(gl.ARRAY_BUFFER, covABuffer);
			gl.bufferData(gl.ARRAY_BUFFER, covA, gl.DYNAMIC_DRAW);

			gl.bindBuffer(gl.ARRAY_BUFFER, covBBuffer);
			gl.bufferData(gl.ARRAY_BUFFER, covB, gl.DYNAMIC_DRAW);

			for (let i = 0; i < shBuffersGl.length; ++i) {
					gl.bindBuffer(gl.ARRAY_BUFFER, shBuffersGl[i]);
					gl.bufferData(gl.ARRAY_BUFFER, shBufferData[`sh${i}`], gl.DYNAMIC_DRAW);
			}
		}
	};

	window.addEventListener(
		"wheel",
		(e) => {
			e.preventDefault();
			viewMatrix = dyncam.getViewMatrix();
		},
		{ passive: false },
	);

	canvas.addEventListener("mousemove", (e) => {
		e.preventDefault();
		viewMatrix = dyncam.getViewMatrix();
	});

	let vertexCount = 0;

	let lastFrame = 0;
	let avgFps = 0;
	let start = 0;
	
  createDatGui(settings => worker.postMessage({ settings }), dyncam);

	const frame = (now) => {
		let actualViewMatrix = dyncam.getViewMatrix();
		const viewProj = multiply4(projectionMatrix, actualViewMatrix);
		worker.postMessage({ view: viewProj });

		const currentFps = 1000 / (now - lastFrame) || 0;
		avgFps = avgFps * 0.9 + currentFps * 0.1;

		if (vertexCount > 0) {
			document.getElementById("spinner").style.display = "none";
			// console.time('render')
			gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
			ext.drawArraysInstancedANGLE(gl.TRIANGLE_FAN, 0, 4, vertexCount);
			// console.timeEnd('render')
		} else {
			gl.clear(gl.COLOR_BUFFER_BIT);
			document.getElementById("spinner").style.display = "";
			start = Date.now() + 2000;
		}
		const progress = (100 * vertexCount) / (splatData.length / rowLength);
		if (progress < 100) {
			document.getElementById("progress").style.width = progress + "%";
		} else {
			document.getElementById("progress").style.display = "none";
		}
		fps.innerText = Math.round(avgFps) + " fps";
		lastFrame = now;
		requestAnimationFrame(frame);
	};

	frame();

	const selectFile = (file) => {
		const fr = new FileReader();
		if (/\.json$/i.test(file.name)) {
			fr.onload = () => {
				cameras = JSON.parse(fr.result);
				viewMatrix = getViewMatrix(cameras[0]);
				projectionMatrix = getProjectionMatrix(
					camera.fx / downsample,
					camera.fy / downsample,
					canvas.width,
					canvas.height,
				);
				gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

				console.log("Loaded Cameras");
			};
			fr.readAsText(file);
		} else {
			stopLoading = true;
			fr.onload = () => {
				splatData = new Uint8Array(fr.result);
				console.log("Loaded", Math.floor(splatData.length / rowLength));

				if (
					splatData[0] == 112 &&
					splatData[1] == 108 &&
					splatData[2] == 121 &&
					splatData[3] == 10
				) {
					// ply file magic header means it should be handled differently
					worker.postMessage({ ply: splatData.buffer });
				} else {
					worker.postMessage({
						buffer: splatData.buffer,
						vertexCount: Math.floor(splatData.length / rowLength),
						sphericalHarmonics
					});
				}
			};
			fr.readAsArrayBuffer(file);
		}
	};

	const preventDefault = (e) => {
		e.preventDefault();
		e.stopPropagation();
	};
	document.addEventListener("dragenter", preventDefault);
	document.addEventListener("dragover", preventDefault);
	document.addEventListener("dragleave", preventDefault);
	document.addEventListener("drop", (e) => {
		e.preventDefault();
		e.stopPropagation();
		selectFile(e.dataTransfer.files[0]);
	});

	window.addEventListener('resize', (e) => {
		const canvas = document.getElementById("canvas");
    	canvas.width = innerWidth / activeDownsample;
    	canvas.height = innerHeight / activeDownsample;
	})

	let bytesRead = 0;
	let lastVertexCount = -1;
	let stopLoading = false;

	while (true) {
		const { done, value } = await reader.read();
		if (done || stopLoading) break;

		splatData.set(value, bytesRead);
		bytesRead += value.length;

		if (vertexCount > lastVertexCount) {
			worker.postMessage({
				buffer: splatData.buffer,
				vertexCount: Math.floor(bytesRead / rowLength),
				sphericalHarmonics
			});
			lastVertexCount = vertexCount;
		}
	}
	if (!stopLoading)
		worker.postMessage({
			buffer: splatData.buffer,
			vertexCount: Math.floor(bytesRead / rowLength),
			sphericalHarmonics
		});
}

main()/*.catch((err) => {
	document.getElementById("spinner").style.display = "none";
	document.getElementById("message").innerText = err.toString();
})*/;

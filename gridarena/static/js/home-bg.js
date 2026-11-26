/* Particle network behind the home page (tef-services ParticleCanvas). */
(function () {
  var canvas = document.getElementById('heroParticles');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');

  var PARTICLE_COUNT = 55;
  var CONNECTION_DISTANCE = 130;
  var PARTICLE_RADIUS = 1.6;
  var LINE_WIDTH = 0.7;
  var SPEED = 0.225;

  var particles = [];
  var colors;

  function readColors() {
    var light = document.documentElement.getAttribute('data-theme') === 'light';
    colors = {
      particle: light ? 'rgba(0, 100, 160, 0.55)' : 'rgba(100, 200, 235, 0.55)',
      line: light ? 'rgba(0, 100, 160,' : 'rgba(100, 200, 235,',
      lineOpacity: light ? 0.3 : 0.35
    };
  }

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function init() {
    particles = [];
    for (var i = 0; i < PARTICLE_COUNT; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * SPEED * 2,
        vy: (Math.random() - 0.5) * SPEED * 2,
        radius: PARTICLE_RADIUS + Math.random()
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    particles.forEach(function (p) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;
    });

    ctx.lineWidth = LINE_WIDTH;
    for (var i = 0; i < particles.length; i++) {
      for (var j = i + 1; j < particles.length; j++) {
        var dx = particles[i].x - particles[j].x;
        var dy = particles[i].y - particles[j].y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECTION_DISTANCE) {
          var opacity = (1 - dist / CONNECTION_DISTANCE) * colors.lineOpacity;
          ctx.beginPath();
          ctx.strokeStyle = colors.line + ' ' + opacity.toFixed(3) + ')';
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    ctx.fillStyle = colors.particle;
    particles.forEach(function (p) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', function () { resize(); init(); });
  document.addEventListener('gridarena:themechange', readColors);

  readColors();
  resize();
  init();
  draw();
})();

document.addEventListener("DOMContentLoaded", () => {
  // Generate sample points representing recorded telemetry
  const sampleFrames = Array.from({ length: 40 }, (_, i) => i * 20);
  const sampleLipWidths = sampleFrames.map(f => +(22 + Math.sin(f / 18) * 3 + Math.random() * 0.8).toFixed(1));
  const sampleEyeDists = sampleFrames.map(f => +(26 + Math.cos(f / 22) * 2 + Math.random() * 0.6).toFixed(1));
  const sampleEarSpans = sampleFrames.map(f => +(66.5 + Math.sin(f / 25) * 4 + Math.random() * 1.2).toFixed(1));

  // Trajectory coordinates
  const leftEarPts = sampleFrames.map(f => ({ x: +(277 + Math.sin(f / 20) * 4).toFixed(1), y: +(113 + Math.cos(f / 15) * 3).toFixed(1) }));
  const rightEarPts = sampleFrames.map(f => ({ x: +(344 + Math.sin(f / 20) * 4).toFixed(1), y: +(107 + Math.cos(f / 15) * 3).toFixed(1) }));
  const leftEyePts = sampleFrames.map(f => ({ x: +(297 + Math.sin(f / 20) * 4).toFixed(1), y: +(106 + Math.cos(f / 15) * 3).toFixed(1) }));
  const rightEyePts = sampleFrames.map(f => ({ x: +(323 + Math.sin(f / 20) * 4).toFixed(1), y: +(103 + Math.cos(f / 15) * 3).toFixed(1) }));
  const nosePts = sampleFrames.map(f => ({ x: +(310 + Math.sin(f / 20) * 4).toFixed(1), y: +(118 + Math.cos(f / 15) * 3).toFixed(1) }));
  const lipPts = sampleFrames.map(f => ({ x: +(314 + Math.sin(f / 20) * 4).toFixed(1), y: +(140 + Math.cos(f / 15) * 3).toFixed(1) }));

  // Initialize Distance & Span Metrics Chart
  const ctxWidth = document.getElementById("lipWidthChart").getContext("2d");
  new Chart(ctxWidth, {
    type: "line",
    data: {
      labels: sampleFrames,
      datasets: [
        {
          label: "Ear Span (px)",
          data: sampleEarSpans,
          borderColor: "#ffb400",
          backgroundColor: "rgba(255, 180, 0, 0.08)",
          fill: true,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2
        },
        {
          label: "Inter-Eye Distance (px)",
          data: sampleEyeDists,
          borderColor: "#00e5ff",
          backgroundColor: "rgba(0, 229, 255, 0.08)",
          fill: true,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2
        },
        {
          label: "Lip Width (px)",
          data: sampleLipWidths,
          borderColor: "#ff4081",
          backgroundColor: "rgba(255, 64, 129, 0.08)",
          fill: true,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#c9d1d9" } }
      },
      scales: {
        x: {
          title: { display: true, text: "Frame Index", color: "#8b949e" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#8b949e" }
        },
        y: {
          title: { display: true, text: "Distance (Pixels)", color: "#8b949e" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#8b949e" }
        }
      }
    }
  });

  // Initialize Spatial Trajectory Scatter Chart
  const ctxTraj = document.getElementById("lipTrajectoryChart").getContext("2d");
  new Chart(ctxTraj, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Left & Right Ears",
          data: [...leftEarPts, ...rightEarPts],
          backgroundColor: "#ffb400",
          pointRadius: 4
        },
        {
          label: "Left & Right Eyes",
          data: [...leftEyePts, ...rightEyePts],
          backgroundColor: "#00e5ff",
          pointRadius: 4
        },
        {
          label: "Nose",
          data: nosePts,
          backgroundColor: "#e040fb",
          pointRadius: 5
        },
        {
          label: "Lips Center",
          data: lipPts,
          backgroundColor: "#ff4081",
          pointRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#c9d1d9" } }
      },
      scales: {
        x: {
          title: { display: true, text: "X Pixel Coordinate", color: "#8b949e" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#8b949e" }
        },
        y: {
          title: { display: true, text: "Y Pixel Coordinate (Inverted)", color: "#8b949e" },
          reverse: true,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#8b949e" }
        }
      }
    }
  });
});

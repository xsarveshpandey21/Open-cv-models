document.addEventListener("DOMContentLoaded", () => {
  // Update GitHub Link if available
  const githubBtn = document.getElementById("github-link-btn");

  // Sample data fallback for Chart.js telemetry charts
  const sampleFrames = Array.from({ length: 50 }, (_, i) => i * 10);
  const sampleLipWidths = sampleFrames.map(f => 45 + Math.sin(f / 15) * 6 + Math.random() * 2);
  const sampleCenterX = sampleFrames.map(f => 310 + Math.cos(f / 20) * 8);
  const sampleCenterY = sampleFrames.map(f => 230 + Math.sin(f / 10) * 5);

  // Initialize Lip Width Chart
  const ctxWidth = document.getElementById("lipWidthChart").getContext("2d");
  new Chart(ctxWidth, {
    type: "line",
    data: {
      labels: sampleFrames,
      datasets: [{
        label: "Lip Width (px)",
        data: sampleLipWidths,
        borderColor: "#58a6ff",
        backgroundColor: "rgba(88, 166, 255, 0.1)",
        fill: true,
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#c9d1d9" } }
      },
      scales: {
        x: {
          title: { display: true, text: "Frame Number", color: "#8b949e" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#8b949e" }
        },
        y: {
          title: { display: true, text: "Width (Pixels)", color: "#8b949e" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#8b949e" }
        }
      }
    }
  });

  // Initialize Trajectory Chart
  const ctxTraj = document.getElementById("lipTrajectoryChart").getContext("2d");
  new Chart(ctxTraj, {
    type: "scatter",
    data: {
      datasets: [{
        label: "Lip Center Point (X, Y)",
        data: sampleCenterX.map((x, i) => ({ x: x, y: sampleCenterY[i] })),
        backgroundColor: "#bc8cff",
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#c9d1d9" } }
      },
      scales: {
        x: {
          title: { display: true, text: "Center X Position", color: "#8b949e" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#8b949e" }
        },
        y: {
          title: { display: true, text: "Center Y Position", color: "#8b949e" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#8b949e" }
        }
      }
    }
  });
});

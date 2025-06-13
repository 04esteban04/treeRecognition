// === Upload Form Submission ===
document.getElementById('upload-form')?.addEventListener('submit', async function (e) {
  e.preventDefault();
  const formData = new FormData(this);
  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    displayResults(data);
  } catch (error) {
    console.error('Upload error:', error);
  }
});

// === Use Default Button ===
document.getElementById('use-default')?.addEventListener('click', async () => {
  try {
    const response = await fetch('/default-images');
    const data = await response.json();
    displayResults(data);
  } catch (error) {
    console.error('Default fetch error:', error);
  }
});

// === Results Display ===
function displayResults(data) {
  const resultsDiv = document.getElementById('results');
  if (!resultsDiv) return;

  resultsDiv.innerHTML = '';
  data.forEach(entry => {
    const div = document.createElement('div');
    div.innerHTML = `
      <img src="/static/uploads/${entry.filename}" alt="${entry.result.species}" />
      <p><strong>Especie:</strong> ${entry.result.species}</p>
      <p><strong>Confianza:</strong> ${entry.result.confidence}</p>
    `;
    resultsDiv.appendChild(div);
  });
}

// === Tab Navigation ===
document.addEventListener("DOMContentLoaded", () => {
  const tabLinks = document.querySelectorAll(".tab-trigger");
  const contentDiv = document.getElementById("mainContent");

  async function loadTabContent(id) {
    try {
      const response = await fetch(`/tab/${id.replace("#", "")}`);
      const html = await response.text();
      contentDiv.innerHTML = html;

      // Update history and active tab styling
      history.pushState(null, "", id);
      tabLinks.forEach(link => link.classList.remove("active"));
      document.querySelector(`.tab-trigger[href="${id}"]`)?.classList.add("active");
    } catch (error) {
      console.error(`Error loading tab ${id}:`, error);
    }
  }

  tabLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = link.getAttribute("href");
      loadTabContent(targetId);
    });
  });

  // On initial load
  const initialTab = window.location.hash || "#home";
  loadTabContent(initialTab);
});

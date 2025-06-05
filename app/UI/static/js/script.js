document.getElementById('upload-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const formData = new FormData(this);

  const response = await fetch('/upload', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  showResults(data);
});

document.getElementById('use-default').addEventListener('click', async () => {
  const response = await fetch('/default-images');
  const data = await response.json();
  showResults(data);
});

function showResults(data) {
  const resultsDiv = document.getElementById('results');
  resultsDiv.innerHTML = '';

  data.forEach(entry => {
    const div = document.createElement('div');
    div.innerHTML = `
      <img src="/static/uploads/${entry.filename}" />
      <p><strong>Especie:</strong> ${entry.result.species}</p>
      <p><strong>Confianza:</strong> ${entry.result.confidence}</p>
    `;
    resultsDiv.appendChild(div);
  });
}

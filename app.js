const checkBtn = document.getElementById('checkBtn');
const urlInput = document.getElementById('urlInput');
const resultDiv = document.getElementById('result');


checkBtn.addEventListener('click', async () => {
const url = urlInput.value.trim();
if (!url) {
resultDiv.innerHTML = '<div class="bad">Please enter a URL</div>';
return;
}


resultDiv.innerHTML = 'Checking...';


try {
const resp = await fetch('/predict', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({ url })
});


const data = await resp.json();
if (!resp.ok) throw new Error(data.error || 'Unknown error');


if (data.label === 'phishing'){
resultDiv.innerHTML = '<div class="bad">❌ Phishing</div>';
} else {
resultDiv.innerHTML = '<div class="ok">✅ Legitimate</div>';
}
} catch (err) {
resultDiv.innerHTML = `<div class="bad">Error: ${err.message}</div>`;
}
});
const header = document.querySelector('[data-header]');
const copyButton = document.querySelector('[data-copy]');

const updateHeader = () => header?.classList.toggle('is-scrolled', window.scrollY > 24);
updateHeader();
window.addEventListener('scroll', updateHeader, { passive: true });

copyButton?.addEventListener('click', async () => {
  const target = document.querySelector(copyButton.dataset.copy);
  const text = target?.innerText.trim();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = 'Copied';
    window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1500);
  } catch {
    copyButton.textContent = 'Select';
  }
});

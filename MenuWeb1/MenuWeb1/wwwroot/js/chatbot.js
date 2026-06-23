const chatWindow = document.getElementById('chatWindow');
const chatToggle = document.getElementById('chatToggle');
const chatClose = document.getElementById('chatClose');
const chatIcon = document.getElementById('chatIcon');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const chatSend = document.getElementById('chatSend');
const typingEl = document.getElementById('typingIndicator');

let selectedAllergens = [];
let isOpen = false;
let conversationHistory = [];

// Chatbot aç/kapat
chatToggle.addEventListener('click', () => {
    isOpen = !isOpen;
    if (isOpen) {
        chatWindow.classList.remove('opacity-0', 'scale-95', 'pointer-events-none');
        chatWindow.classList.add('opacity-100', 'scale-100');
        chatIcon.textContent = 'close';
    } else {
        chatWindow.classList.add('opacity-0', 'scale-95', 'pointer-events-none');
        chatWindow.classList.remove('opacity-100', 'scale-100');
        chatIcon.textContent = 'chat';
    }
});

chatClose.addEventListener('click', () => chatToggle.click());

// Alerjen seçimi
document.querySelectorAll('.allergen-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const allergen = btn.dataset.allergen;
        btn.classList.toggle('bg-primary');
        btn.classList.toggle('text-white');
        btn.classList.toggle('border-primary');
        if (selectedAllergens.includes(allergen)) {
            selectedAllergens = selectedAllergens.filter(a => a !== allergen);
        } else {
            selectedAllergens.push(allergen);
        }
        // Alerjen değişince konuşma geçmişini sıfırla
        conversationHistory = [];
    });
});

function addMessage(role, text) {
    const wrap = document.createElement('div');
    wrap.className = `flex flex-col ${role === 'user' ? 'items-end' : 'items-start'}`;
    if (role === 'assistant') {
        wrap.innerHTML = `
            <span class="text-xs font-semibold text-primary mb-1">Asistan</span>
            <div class="bg-surface-container rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-on-surface max-w-[85%] shadow-sm">${text}</div>`;
    } else {
        wrap.innerHTML = `
            <div class="bg-primary rounded-2xl rounded-tr-sm px-4 py-3 text-sm text-on-primary max-w-[85%]">${text}</div>`;
    }
    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addWarning(warnings) {
    const limited = warnings.slice(0, 5);
    const extra = warnings.length - 5;
    const div = document.createElement('div');
    div.className = 'bg-yellow-50 border border-yellow-300 rounded-xl px-4 py-3 text-sm text-yellow-800';
    div.innerHTML = `<strong class="block mb-1">⚠️ Alerjen Uyarısı</strong>
        ${limited.map(w => `• ${w}`).join('<br>')}
        ${extra > 0 ? `<br><span class="text-xs">...ve ${extra} ürün daha</span>` : ''}`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Kullanıcı mesajını geçmişe ekle
    conversationHistory.push({ role: 'user', content: query });

    addMessage('user', query);
    chatInput.value = '';
    typingEl.classList.remove('hidden');
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const res = await fetch('http://localhost:5000/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                allergens: selectedAllergens,
                history: conversationHistory,
                city: window.selectedCity || null
            })
        });
        const data = await res.json();
        typingEl.classList.add('hidden');
        if (data.warnings && data.warnings.length > 0) addWarning(data.warnings);
        addMessage('assistant', data.answer);

        // Asistan yanıtını geçmişe ekle
        conversationHistory.push({ role: 'assistant', content: data.answer });

    } catch {
        typingEl.classList.add('hidden');
        addMessage('assistant', 'Bağlantı hatası. Flask API çalışıyor mu?');
    }
}

chatSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });cd
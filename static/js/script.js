// Mobile menu toggle
document.addEventListener('DOMContentLoaded', function() {
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', function() {
            const isVisible = navLinks.style.display === 'flex';
            
            if (isVisible) {
                navLinks.style.display = 'none';
            } else {
                navLinks.style.display = 'flex';
                
                // Mobile styles
                if (window.innerWidth <= 768) {
                    navLinks.style.flexDirection = 'column';
                    navLinks.style.position = 'absolute';
                    navLinks.style.top = '100%';
                    navLinks.style.left = '0';
                    navLinks.style.right = '0';
                    navLinks.style.backgroundColor = 'white';
                    navLinks.style.padding = '20px';
                    navLinks.style.boxShadow = '0 5px 10px rgba(0,0,0,0.1)';
                }
            }
        });
    }

    // Demo email analysis
    window.analyzeEmail = function() {
        const emailContent = document.getElementById('emailContent').value;
        const resultCard = document.querySelector('.result-card');
        
        if (!emailContent.trim()) {
            alert('Please enter some email content to analyze.');
            return;
        }
        
        // Show loading state
        resultCard.innerHTML = '<div style="text-align: center; padding: 40px;">Analyzing email content...</div>';
        
        // Simulate API call
        setTimeout(() => {
            const spamProbability = Math.random() * 100;
            const isSpam = spamProbability > 70;
            
            resultCard.innerHTML = `
                <div class="result-header">
                    <h4>Spam Probability</h4>
                    <span class="spam-score ${isSpam ? 'high' : 'low'}">${Math.round(spamProbability)}%</span>
                </div>
                <div class="result-details">
                    <div class="risk-factor">
                        <span class="factor-name">Suspicious Keywords</span>
                        <span class="factor-risk ${isSpam ? 'high' : 'low'}">${isSpam ? 'High Risk' : 'Low Risk'}</span>
                    </div>
                    <div class="risk-factor">
                        <span class="factor-name">Sender Reputation</span>
                        <span class="factor-risk medium">Medium Risk</span>
                    </div>
                    <div class="risk-factor">
                        <span class="factor-name">Content Analysis</span>
                        <span class="factor-risk ${isSpam ? 'high' : 'low'}">${isSpam ? 'High Risk' : 'Low Risk'}</span>
                    </div>
                </div>
                <div class="verdict" style="background: ${isSpam ? '#f8d7da' : '#d4edda'}; color: ${isSpam ? '#721c24' : '#155724'};">
                    <i class="fas fa-${isSpam ? 'exclamation-triangle' : 'check-circle'}"></i>
                    <strong>This email is ${isSpam ? 'likely SPAM' : 'probably SAFE'}</strong>
                </div>
            `;
        }, 1500);
    };
});

// Close mobile menu when clicking outside
document.addEventListener('click', function(e) {
    const navLinks = document.querySelector('.nav-links');
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    
    if (window.innerWidth <= 768 && 
        navLinks && navLinks.style.display === 'flex' &&
        !e.target.closest('.navbar-container')) {
        navLinks.style.display = 'none';
    }
});

// Replace the setTimeout block with this:
fetch('/api/detect-spam', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        email: email,
        emailBody: emailBody
    })
})
.then(response => response.json())
.then(data => {
    if (data.isSpam) {
        resultDiv.textContent = '⚠️ This email is SPAM';
        resultDiv.className = 'result spam';
    } else {
        resultDiv.textContent = '✅ This email is SAFE';
        resultDiv.className = 'result safe';
    }
    resultDiv.style.display = 'block';
})
.catch(error => {
    resultDiv.textContent = '❌ Error analyzing email';
    resultDiv.className = 'result spam';
    resultDiv.style.display = 'block';
})
.finally(() => {
    loadingDiv.style.display = 'none';
    detectButton.disabled = false;
    detectButton.textContent = 'Detect Email';
});
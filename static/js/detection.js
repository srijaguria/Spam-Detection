document.addEventListener('DOMContentLoaded', function() {
    // Add focus effects
    var inputs = document.querySelectorAll('.detection-input, .detection-textarea');
    inputs.forEach(function(input) {
        input.addEventListener('focus', function() {
            this.style.transform = 'translateY(-2px)';
        });
        input.addEventListener('blur', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Animate confidence bar if result exists
    var confidenceFill = document.querySelector('.confidence-fill');
    if (confidenceFill) {
        var width = parseInt(confidenceFill.style.width);
        setTimeout(function() {
            confidenceFill.style.width = width + '%';
        }, 100);
    }
});
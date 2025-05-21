function showAnalysisPage() {
    window.location.href="/analysis"
}

document.getElementById('stockForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const ticker = document.getElementById('ticker').value.toUpperCase();
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');
    const results = document.getElementById('results');
    
    loadingSpinner.classList.remove('d-none');
    errorMessage.classList.add('d-none');
    results.classList.add('d-none');
    
    try {
        const formData = new FormData();
        formData.append('ticker', ticker);
        
        console.log('Sending request for ticker:', ticker); // Debugging log
        
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        console.log('Response status:', response.status); // Debugging log
        
        const data = await response.json();
        
        console.log('Received data:', data); // Debugging log
        
        if (!response.ok) {
            throw new Error(data.error || 'An unknown error occurred');
        }
        
        // Rest of your existing code...
        
    } catch (error) {
        console.error('Full error details:', error);
        errorMessage.textContent = `Error: ${error.message}`;
        errorMessage.classList.remove('d-none');
    } finally {
        loadingSpinner.classList.add('d-none');
    }
});
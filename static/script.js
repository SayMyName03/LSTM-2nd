function showAnalysisPage() {
    document.getElementById('landing-page').style.display = 'none';
    document.getElementById('analysis-page').style.display = 'block';
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
        
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update metrics
        document.getElementById('currentPrice').textContent = `$${data.current_price.toFixed(2)}`;
        document.getElementById('volume').textContent = data.volume.toLocaleString();
        document.getElementById('high52').textContent = `$${data.high_52week.toFixed(2)}`;
        document.getElementById('low52').textContent = `$${data.low_52week.toFixed(2)}`;
        document.getElementById('predictionValue').textContent = `$${data.prediction.toFixed(2)}`;
        
        // Create price chart
        const trace = {
            type: 'scatter',
            x: data.dates,
            y: data.closing_prices,
            name: 'Stock Price',
            line: {
                color: 'rgba(0, 123, 255, 0.8)',
                width: 2
            }
        };
        
        const layout = {
            title: {
                text: `${ticker} Stock Price`,
                font: {
                    color: 'white'
                }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            yaxis: {
                title: 'Price',
                gridcolor: 'rgba(255,255,255,0.1)',
                color: 'white'
            },
            xaxis: {
                title: 'Date',
                gridcolor: 'rgba(255,255,255,0.1)',
                color: 'white'
            },
            height: 500,
            margin: {
                l: 50,
                r: 20,
                t: 40,
                b: 50
            }
        };
        
        Plotly.newPlot('priceChart', [trace], layout);
        
        results.classList.remove('d-none');
        
    } catch (error) {
        errorMessage.textContent = error.message;
        errorMessage.classList.remove('d-none');
    } finally {
        loadingSpinner.classList.add('d-none');
    }
});
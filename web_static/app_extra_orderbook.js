
// ==================== Order Book Logic ====================

let orderBookChart = null;
let orderBookInterval = null;
let orderBookInFlight = false;
let orderBookUpdateTimer = null;

async function fetchJsonWithTimeout(url, options = {}, cfg = {}) {
    const {
        timeoutMs = 8000,
        retries = 1,
        retryDelayMs = 600
    } = cfg || {};

    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const resp = await fetch(url, { ...options, signal: controller.signal });
            clearTimeout(timer);
            if (!resp.ok) {
                if (attempt < retries) {
                    await new Promise(r => setTimeout(r, retryDelayMs * (attempt + 1)));
                    continue;
                }
                throw new Error(`HTTP ${resp.status}`);
            }
            return await resp.json();
        } catch (err) {
            clearTimeout(timer);
            lastErr = err;
            if (attempt < retries) {
                await new Promise(r => setTimeout(r, retryDelayMs * (attempt + 1)));
                continue;
            }
        }
    }
    throw lastErr || new Error('network_error');
}

async function fetchOrderBook() {
    const symbol = document.getElementById('orderbook-symbol').value;
    const exchange = appState.selectedExchange || 'binance';

    if (orderBookInFlight) {
        return;
    }
    orderBookInFlight = true;
    
    try {
        const result = await fetchJsonWithTimeout(
            `/api/spot/orderbook?exchange=${exchange}&symbol=${encodeURIComponent(symbol)}&limit=20`,
            {},
            { timeoutMs: 6000, retries: 1 }
        );
        
        if (result.success && result.data) {
            renderOrderBook(result.data);
            addLog(`盘口数据更新: ${symbol} (${exchange})`, 'success');
        } else {
            console.error('Fetch OrderBook failed:', result.error);
            addLog(`盘口数据获取失败: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Fetch OrderBook error:', error);
    } finally {
        orderBookInFlight = false;
    }
}

function renderOrderBook(data) {
    const bidsTable = document.getElementById('bids-table').querySelector('tbody');
    const asksTable = document.getElementById('asks-table').querySelector('tbody');
    
    // Clear existing
    bidsTable.innerHTML = '';
    asksTable.innerHTML = '';
    
    // Render Bids
    const bidRows = data.bids.map(([price, amount]) => {
        const p = Number(price);
        const a = Number(amount);
        const total = Number.isFinite(p * a) ? (p * a).toFixed(2) : '-';
        return `<tr>
            <td class="text-success">${price}</td>
            <td>${amount}</td>
            <td>${total}</td>
        </tr>`;
    }).join('');
    bidsTable.innerHTML = bidRows;
    
    // Render Asks (Reversed to show best ask at bottom if desired, but standard is sorted by price asc)
    // Asks come sorted by price ascending (lowest ask first).
    const askRows = data.asks.map(([price, amount]) => {
        const p = Number(price);
        const a = Number(amount);
        const total = Number.isFinite(p * a) ? (p * a).toFixed(2) : '-';
        return `<tr>
            <td class="text-danger">${price}</td>
            <td>${amount}</td>
            <td>${total}</td>
        </tr>`;
    }).join('');
    asksTable.innerHTML = askRows;

    // Render Chart
    renderOrderBookChart(data);
}

function renderOrderBookChart(data) {
    const ctx = document.getElementById('orderbook-chart').getContext('2d');
    
    // Sort asks descending for display (left to right) if needed, but stepped charts usually work best with raw price
    // Bids should be high to low (best bid first). Data comes raw [price, amount]
    
    // Prepare data points
    const bids = data.bids.map(x => ({x: x[0], y: x[1]}));
    const asks = data.asks.map(x => ({x: x[0], y: x[1]}));

    if (orderBookChart) {
        // Update existing chart data to prevent flickering
        orderBookChart.data.datasets[0].data = bids;
        orderBookChart.data.datasets[1].data = asks;
        if (!orderBookUpdateTimer) {
            orderBookUpdateTimer = requestAnimationFrame(() => {
                orderBookUpdateTimer = null;
                try {
                    orderBookChart.update('none'); // 'none' mode prevents animation on every update for performance
                } catch {
                    // ignore
                }
            });
        }
    } else {
        // Create new chart
        orderBookChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Bids',
                    data: bids,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    fill: true,
                    stepped: true
                }, {
                    label: 'Asks',
                    data: asks,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.2)',
                    fill: true,
                    stepped: true
                }]
            },
            options: {
                responsive: true,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                animation: {
                    duration: 0 // Disable initial animation for real-time feel
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Price' }
                    },
                    y: {
                        title: { display: true, text: 'Volume' }
                    }
                }
            }
        });
    }
}

// Event Listeners for Order Book
document.addEventListener('DOMContentLoaded', () => {
    // ... existing listeners ...
    
    const fetchBtn = document.getElementById('btn-fetch-orderbook');
    if (fetchBtn) {
        fetchBtn.addEventListener('click', fetchOrderBook);
    }
    
    const autoRefreshSwitch = document.getElementById('auto-refresh-switch');
    if (autoRefreshSwitch) {
        autoRefreshSwitch.addEventListener('change', (e) => {
            if (e.target.checked) {
                fetchOrderBook(); // Fetch immediately
                orderBookInterval = setInterval(fetchOrderBook, 2000); // Poll every 2s
            } else {
                if (orderBookInterval) {
                    clearInterval(orderBookInterval);
                    orderBookInterval = null;
                }
            }
        });
    }
});

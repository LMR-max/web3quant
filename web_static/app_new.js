/**
 * 加密货币数据系统 - 完整前端应用
 * 支持完整的数据获取、分析、管理功能
 */

// ==================== 全局状态 ====================

const appState = {
    config: {},
    selectedExchange: 'binance',
    selectedMarket: 'spot',
    selectedTimeframe: '1h',
    selectedSpotDataType: 'ohlcv',
    selectedOptionDataType: 'ohlcv',
    selectedMarginDataType: 'ohlcv',
    selectedDerivDataType: 'ohlcv',
    selectedSymbols: [],
    fetchedData: {},
    dateRange: null,
    systemLogs: [],

    // AlphaGen / RL
    alphagenJobId: null,
    alphagenPollTimer: null,

    // Metadata Cache
    operators: [],
    columns: [],
    windows: [],

    // ML
    mlTaskId: null,
    mlPollTimer: null,

    // Onchain
    onchainResults: {}
};

// ==================== AlphaGen Meta Helpers ====================
async function loadAlphaGenMeta() {
    const panelPathEl = document.getElementById('alphagen-panel');
    if (!panelPathEl) return;
    const panelPath = panelPathEl.value;
    if (!panelPath) return;

    try {
        const response = await fetch(`/api/alphagen/meta?path=${encodeURIComponent(panelPath)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data) {
            appState.operators = data.operators || [];
            appState.columns = data.columns || [];
            appState.windows = data.windows || [];

            // Render for both sections (Prep and Eval)
            renderAlphaGenMeta('prep');
            renderAlphaGenMeta('eval');

            const metaContainerPrep = document.getElementById('dsl-meta-container-prep');
            if (metaContainerPrep) metaContainerPrep.style.display = '';

            const metaContainerEval = document.getElementById('dsl-meta-container-eval');
            if (metaContainerEval) metaContainerEval.style.display = '';
        }
    } catch (err) {
        console.error("Failed to load metadata", err);
    }
}

function renderAlphaGenMeta(suffix) {
    // Suffix is 'prep' or 'eval'
    if (!suffix) suffix = 'prep'; // fallback

    // Render Columns
    const colContainer = document.getElementById(`meta-cols-list-${suffix}`);
    if (!colContainer) return;
    colContainer.innerHTML = '';

    const colCountEl = document.getElementById(`meta-col-count-${suffix}`);
    if (colCountEl) colCountEl.textContent = appState.columns.length;

    if (appState.columns.length === 0) {
        colContainer.innerHTML = '<span class="text-muted">No columns found. Check file path.</span>';
    } else {
        appState.columns.forEach(col => {
            const badge = document.createElement('span');
            badge.className = 'badge bg-white text-dark border me-1 mb-1';
            badge.style.cursor = 'pointer';
            badge.style.fontWeight = 'normal';
            badge.textContent = col;
            badge.addEventListener('click', () => insertToExpr(col));
            colContainer.appendChild(badge);
        });
    }

    // Render Ops
    const opContainer = document.getElementById(`meta-ops-list-${suffix}`);
    if (!opContainer) return;
    opContainer.innerHTML = '';

    if (appState.operators.length === 0) {
        opContainer.innerHTML = '<span class="text-muted">No operators loaded.</span>';
    } else {
        appState.operators.forEach(op => {
            let title = op.name;
            if (op.arg_kinds) title += `(${op.arg_kinds.join(', ')})`;

            // Color code based on type
            let colorClass = 'bg-light text-dark border';
            if (op.name.startsWith('ts_')) colorClass = 'bg-info bg-opacity-10 text-dark border border-info';
            if (op.name.startsWith('cs_')) colorClass = 'bg-success bg-opacity-10 text-dark border border-success';
            if (['if_else', 'is_jump'].includes(op.name)) colorClass = 'bg-warning bg-opacity-10 text-dark border border-warning';

            const badge = document.createElement('span');
            badge.className = `badge ${colorClass} me-1 mb-1`;
            badge.style.cursor = 'pointer';
            badge.style.fontWeight = 'normal';
            badge.title = title;
            badge.textContent = op.name;
            badge.addEventListener('click', () => insertToExpr(op.name + '('));
            opContainer.appendChild(badge);
        });

        // Add windows hint
        const hint = document.createElement('div');
        hint.className = 'mt-2 text-muted small';
        hint.textContent = `Windows: ${appState.windows.join(', ')}`;
        opContainer.appendChild(hint);
    }
}

function insertToExpr(text) {
    const txtEl = document.getElementById('alphagen-expr');
    if (!txtEl) return;

    const curVal = txtEl.value;
    const startPos = txtEl.selectionStart || 0;
    const endPos = txtEl.selectionEnd || 0;

    txtEl.value = curVal.substring(0, startPos) + text + curVal.substring(endPos, curVal.length);
    txtEl.selectionStart = startPos + text.length;
    txtEl.selectionEnd = startPos + text.length;
    txtEl.focus();
}

// Hook into panel input change (wait for DOM ready)
function setupAlphaGenMeta() {
    const panelEl = document.getElementById('alphagen-panel');
    if (panelEl) {
        panelEl.addEventListener('change', loadAlphaGenMeta);
        // Initial load if value exists
        if (panelEl.value) {
            setTimeout(loadAlphaGenMeta, 500);
        }
    }

    const evalModeEl = document.getElementById('alphagen-eval-mode');
    if (evalModeEl) {
        evalModeEl.addEventListener('change', function () {
            const hintEl = document.getElementById('multi-asset-hint');
            if (!hintEl) return;
            if (this.value === 'multi') {
                hintEl.style.display = '';
            } else {
                hintEl.style.display = 'none';
            }
        });
    }
}

// 数据分析图表实例
let analysisCharts = {
    price: null,
    volume: null,
    rsi: null
};

// 数据分析-指标显示开关（最小化：只控制当前已实现的叠加/子图）
let analysisIndicatorState = {
    sma20: true,
    sma50: true,
    boll: true,
    rsi: true,
    takerBuyPct: true,
    ema12: true,
    ema26: true,
    vwap: true,
    macd: true,
    vwma20: true,
    hl2: false,
    signals: true,
    kdj: true,
    cci: true,
    bbwidth: true,
    atr: true
};

function readAnalysisIndicatorStateFromUI() {
    const getChecked = (id, fallback) => {
        const el = document.getElementById(id);
        return el ? !!el.checked : !!fallback;
    };
    analysisIndicatorState = {
        sma20: getChecked('ind-sma20', analysisIndicatorState.sma20),
        sma50: getChecked('ind-sma50', analysisIndicatorState.sma50),
        boll: getChecked('ind-boll', analysisIndicatorState.boll),
        rsi: getChecked('ind-rsi', analysisIndicatorState.rsi),
        takerBuyPct: getChecked('ind-takerbuy', analysisIndicatorState.takerBuyPct),
        ema12: getChecked('ind-ema12', analysisIndicatorState.ema12),
        ema26: getChecked('ind-ema26', analysisIndicatorState.ema26),
        vwap: getChecked('ind-vwap', analysisIndicatorState.vwap),
        macd: getChecked('ind-macd', analysisIndicatorState.macd),
        vwma20: getChecked('ind-vwma20', analysisIndicatorState.vwma20),
        hl2: getChecked('ind-hl2', analysisIndicatorState.hl2),
        signals: getChecked('ind-signals', analysisIndicatorState.signals),
        kdj: getChecked('ind-kdj', analysisIndicatorState.kdj),
        cci: getChecked('ind-cci', analysisIndicatorState.cci),
        bbwidth: getChecked('ind-bbwidth', analysisIndicatorState.bbwidth),
        atr: getChecked('ind-atr', analysisIndicatorState.atr)
    };
    return analysisIndicatorState;
}

function applyAnalysisIndicatorStateToUI() {
    const setChecked = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.checked = !!val;
    };
    setChecked('ind-sma20', analysisIndicatorState.sma20);
    setChecked('ind-sma50', analysisIndicatorState.sma50);
    setChecked('ind-boll', analysisIndicatorState.boll);
    setChecked('ind-rsi', analysisIndicatorState.rsi);
    setChecked('ind-takerbuy', analysisIndicatorState.takerBuyPct);
    setChecked('ind-ema12', analysisIndicatorState.ema12);
    setChecked('ind-ema26', analysisIndicatorState.ema26);
    setChecked('ind-vwap', analysisIndicatorState.vwap);
    setChecked('ind-macd', analysisIndicatorState.macd);
    setChecked('ind-vwma20', analysisIndicatorState.vwma20);
    setChecked('ind-hl2', analysisIndicatorState.hl2);
    setChecked('ind-signals', analysisIndicatorState.signals);
    setChecked('ind-kdj', analysisIndicatorState.kdj);
    setChecked('ind-cci', analysisIndicatorState.cci);
    setChecked('ind-bbwidth', analysisIndicatorState.bbwidth);
    setChecked('ind-atr', analysisIndicatorState.atr);
}

const ONCHAIN_DUNE_TEMPLATES = {
    exchange_flow: { dune_params: { hours: 24 }, note: '填写你的交易所资金流 Query ID' },
    address_behavior: { dune_params: { window_hours: 24, retention_hours: 24 }, note: '填写活跃/新增/留存 Query ID' },
    large_moves: { dune_params: { hours: 24, min_value: 100 }, note: '填写大额转账 Query ID' },
    mev: { dune_params: { hours: 24 }, note: '填写 MEV Query ID' },
    gas: { dune_params: { hours: 1 }, note: '填写 Gas Query ID' },
    protocol: { dune_params: { days: 7 }, note: '填写 DEX/TVL Query ID' },
    capital_cycle: { dune_params: { hours: 24 }, note: '填写资金循环 Query ID' },
    token_distribution: { dune_params: { top_n: 20 }, note: '填写持币分布 Query ID' },
    nft: { dune_params: { days: 7 }, note: '填写 NFT Query ID' },
    price_relation: { dune_params: { days: 30 }, metric_key: 'metric', note: '填写价格关联 Query ID' }
};

const ONCHAIN_SUBGRAPH_TEMPLATES = {
    uniswap_v3_ethereum: {
        subgraph: 'uniswap_v3_ethereum',
        query: `query($first:Int){
  pools(first: $first, orderBy: volumeUSD, orderDirection: desc) {
    id
    volumeUSD
    totalValueLockedUSD
  }
}`,
        variables: { first: 10 }
    },
    sushiswap_ethereum: {
        subgraph: 'sushiswap_ethereum',
        query: `query($first:Int){
  pairs(first: $first, orderBy: volumeUSD, orderDirection: desc){
    id
    volumeUSD
    reserveUSD
  }
}`,
        variables: { first: 10 }
    },
    curve_finance: {
        subgraph: 'curve_finance',
        query: `query($first:Int){
  pools(first: $first, orderBy: totalValueLockedUSD, orderDirection: desc){
    id
    totalValueLockedUSD
    volumeUSD
  }
}`,
        variables: { first: 10 }
    }
};

const ONCHAIN_ADDRESS_PRESETS = {
    binance: [
        '0x28C6c06298d514Db089934071355E5743bf21d60',
        '0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549',
        '0xDFd5293D8e347dFe59E90eFd55b2956a1343963d'
    ],
    okx: [
        '0x6cC5F688a315f3dC28A7781717a9A798a59fDA7b',
        '0x236f9f97e0E62388479bf9E5BA4889e46B0273C3'
    ],
    coinbase: [
        '0xA9D1e08C7793af67e9d92fe308d5697FB81d3E43',
        '0x77696bb39917C91A0c3908D577c5B7aD905D1169'
    ],
    kraken: [
        '0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2',
        '0x53d284357ec70cE289D6D64134DfAc8E511c8a3D'
    ]
};

async function copyToClipboard(text) {
    try {
        if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch (e) {
        // ignore and fallback
    }

    try {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        textarea.style.top = '0';
        document.body.appendChild(textarea);
        textarea.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(textarea);
        return !!ok;
    } catch (e) {
        return false;
    }
}

function updateSelectOptions(selectEl, values, preferredValue) {
    if (!selectEl) return;
    const prev = preferredValue ?? selectEl.value;
    selectEl.innerHTML = '';
    (values || []).forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        selectEl.appendChild(opt);
    });
    if (values && values.includes(prev)) {
        selectEl.value = prev;
    } else if (values && values.length > 0) {
        selectEl.value = values[0];
    }
}

function applySupportedTimeframes(allowedTimeframes) {
    const tf = document.getElementById('timeframe');
    if (!tf || !Array.isArray(allowedTimeframes) || allowedTimeframes.length === 0) return;

    const options = Array.from(tf.options || []);
    options.forEach(opt => {
        opt.disabled = !allowedTimeframes.includes(opt.value);
    });

    if (!allowedTimeframes.includes(tf.value)) {
        const firstEnabled = options.find(o => !o.disabled);
        if (firstEnabled) tf.value = firstEnabled.value;
    }
    appState.selectedTimeframe = tf.value;
}

function formatCompactNumber(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    try {
        return new Intl.NumberFormat('zh-CN', {
            notation: 'compact',
            maximumFractionDigits: digits
        }).format(n);
    } catch {
        return n.toFixed(digits);
    }
}

function formatPrice(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    const abs = Math.abs(n);
    if (abs >= 1000) return n.toFixed(2);
    if (abs >= 1) return n.toFixed(4);
    return n.toFixed(6);
}

function updateBrokerHeader(symbol, last, changePct, turnover) {
    const badge = document.getElementById('broker-symbol-badge');
    const lastEl = document.getElementById('broker-last');
    const changeEl = document.getElementById('broker-change');
    const turnEl = document.getElementById('broker-turnover');

    if (badge) badge.textContent = symbol || '--';
    if (lastEl) lastEl.textContent = Number.isFinite(Number(last)) ? formatPrice(last) : '--';

    if (changeEl) {
        changeEl.classList.remove('price-up', 'price-down');
        if (Number.isFinite(changePct)) {
            const pctText = `${(changePct * 100).toFixed(2)}%`;
            changeEl.textContent = pctText;
            changeEl.classList.add(changePct >= 0 ? 'price-up' : 'price-down');
        } else {
            changeEl.textContent = '--';
        }
    }

    if (turnEl) {
        turnEl.textContent = Number.isFinite(Number(turnover)) ? formatCompactNumber(turnover, 2) : '--';
    }
}

function updateBrokerFromOhlcv(symbol, rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        updateBrokerHeader(symbol, NaN, NaN, NaN);
        return;
    }
    const lastRow = rows[rows.length - 1];
    const isArrayRow = Array.isArray(lastRow);
    const open = isArrayRow ? lastRow[1] : lastRow.open;
    const close = isArrayRow ? lastRow[4] : lastRow.close;
    const volume = isArrayRow ? lastRow[5] : (lastRow.quote_volume ?? lastRow.volume);
    const changePct = Number.isFinite(Number(open)) && Number(open) !== 0
        ? (Number(close) - Number(open)) / Number(open)
        : NaN;

    updateBrokerHeader(symbol, close, changePct, volume);
}

function normalizeTimestampMs(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    const abs = Math.abs(n);
    // seconds (10-digit) -> ms
    if (abs < 1e11) return Math.round(n * 1000);
    // milliseconds (13-digit)
    if (abs < 1e14) return Math.round(n);
    // microseconds (16-digit)
    if (abs < 1e17) return Math.round(n / 1000);
    // nanoseconds
    return Math.round(n / 1e6);
}

function resetSymbolsUI() {
    appState.selectedSymbols = [];
    allSymbols = [];
    updateSelectedSymbolsDisplay();
    const search = document.getElementById('symbol-search-input');
    if (search) search.value = '';
    const list = document.getElementById('symbol-list-container');
    if (list) list.style.display = 'none';
    const count = document.getElementById('symbol-count');
    if (count) count.textContent = '点击"加载交易对"按钮获取列表';
}

function resetFetchedDataUI() {
    appState.fetchedData = {};
    const tbody = document.getElementById('fetch-results-tbody');
    if (tbody) tbody.innerHTML = '';
    const stats = document.getElementById('fetch-stats');
    if (stats) stats.innerHTML = '';
}

async function saveLastMergedResultsToServer(results) {
    try {
        const marketLower = (appState.selectedMarket || 'spot').toLowerCase();
        let dataType = 'ohlcv';
        if (marketLower === 'spot') dataType = appState.selectedSpotDataType || 'ohlcv';
        else if (marketLower === 'option') dataType = appState.selectedOptionDataType || 'ohlcv';
        else if (marketLower === 'margin') dataType = appState.selectedMarginDataType || 'ohlcv';
        else if (marketLower === 'swap' || marketLower === 'future') dataType = appState.selectedDerivDataType || 'ohlcv';

        const { startMs, endMs } = getDateRangeMs();

        const meta = {
            exchange: appState.selectedExchange,
            market: appState.selectedMarket,
            timeframe: appState.selectedTimeframe,
            data_type: dataType,
            start_time: Number.isFinite(startMs) ? startMs : null,
            end_time: Number.isFinite(endMs) ? endMs : null
        };

        await fetch('/api/web/merged-results/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: results || {}, meta })
        });
    } catch (e) {
        // 持久化失败不阻塞主流程
        console.warn('保存合并结果失败:', e);
    }
}

async function loadLastMergedResultsFromServer() {
    try {
        const res = await fetch('/api/web/merged-results/load');
        const data = await res.json();
        if (!data || !data.success || !data.exists || !data.payload) return;

        const payload = data.payload;
        const meta = payload.meta || {};
        const results = payload.results;
        if (!results || typeof results !== 'object') return;

        // 尽量恢复选择项（不强制，避免与当前 UI 约束冲突）
        const marketEl = document.getElementById('market');
        const exchangeEl = document.getElementById('exchange');
        const tfEl = document.getElementById('timeframe');
        if (marketEl && meta.market) marketEl.value = meta.market;
        if (exchangeEl && meta.exchange) exchangeEl.value = meta.exchange;
        if (tfEl && meta.timeframe) tfEl.value = meta.timeframe;

        if (meta.market) appState.selectedMarket = meta.market;
        if (meta.exchange) appState.selectedExchange = meta.exchange;
        if (meta.timeframe) appState.selectedTimeframe = meta.timeframe;

        // 先根据 market 应用支持范围
        try {
            await updateMarketUI();
        } catch {
            // ignore
        }

        const marketLower = (appState.selectedMarket || 'spot').toLowerCase();
        if (meta.data_type) {
            if (marketLower === 'spot') {
                const el = document.getElementById('spot-data-type');
                if (el) el.value = meta.data_type;
                appState.selectedSpotDataType = meta.data_type;
            } else if (marketLower === 'option') {
                const el = document.getElementById('option-data-type');
                if (el) el.value = meta.data_type;
                appState.selectedOptionDataType = meta.data_type;
            } else if (marketLower === 'margin') {
                const el = document.getElementById('margin-data-type');
                if (el) el.value = meta.data_type;
                appState.selectedMarginDataType = meta.data_type;
            } else if (marketLower === 'swap' || marketLower === 'future') {
                const el = document.getElementById('deriv-data-type');
                if (el) el.value = meta.data_type;
                appState.selectedDerivDataType = meta.data_type;
            }
        }

        appState.fetchedData = results;
        refreshAnalysisSymbolOptions();
        displayFetchResults(appState.fetchedData);
        addLog('已从磁盘恢复最近一次合并结果（刷新页面仍可用）', 'info');
    } catch (e) {
        console.warn('加载合并结果失败:', e);
    }
}

async function updateMarketUI() {
    const market = (appState.selectedMarket || 'spot').toLowerCase();
    const exchangeSelect = document.getElementById('exchange');
    const spotTypeWrap = document.getElementById('spot-data-type-wrap');
    const spotTypeSelect = document.getElementById('spot-data-type');
    const optionTypeWrap = document.getElementById('option-data-type-wrap');
    const optionTypeSelect = document.getElementById('option-data-type');
    const marginTypeWrap = document.getElementById('margin-data-type-wrap');
    const marginTypeSelect = document.getElementById('margin-data-type');
    const derivTypeWrap = document.getElementById('deriv-data-type-wrap');
    const derivTypeSelect = document.getElementById('deriv-data-type');

    try {
        const res = await fetch(`/api/market-info/${encodeURIComponent(market)}`);
        const payload = await res.json();
        const info = payload && payload.info ? payload.info : null;

        if (info && Array.isArray(info.supported_exchanges) && exchangeSelect) {
            updateSelectOptions(exchangeSelect, info.supported_exchanges, appState.selectedExchange);
            appState.selectedExchange = exchangeSelect.value;
        }
        if (info && Array.isArray(info.timeframes)) {
            applySupportedTimeframes(info.timeframes);
        }
    } catch (e) {
        addLog(`获取市场信息失败: ${e}`, 'warning');
    }

    // onchain/social 将使用“指标/舆情”数据而非 OHLCV；保留按钮可用
    if (market === 'onchain' || market === 'social') {
        addLog(`当前市场(${market})将按指标/舆情模式获取数据`, 'info');
    }

    // 仅 spot 显示“现货数据类型”选择
    if (spotTypeWrap) {
        const isSpot = market === 'spot';
        spotTypeWrap.style.display = isSpot ? '' : 'none';
        if (!isSpot) {
            appState.selectedSpotDataType = 'ohlcv';
            if (spotTypeSelect) spotTypeSelect.value = 'ohlcv';
        }
    }

    // 仅 option 显示“期权数据类型”选择
    if (optionTypeWrap) {
        const isOption = market === 'option';
        optionTypeWrap.style.display = isOption ? '' : 'none';
        if (!isOption) {
            appState.selectedOptionDataType = 'ohlcv';
            if (optionTypeSelect) optionTypeSelect.value = 'ohlcv';
        }
    }

    // 仅 margin 显示“杠杆数据类型”选择
    if (marginTypeWrap) {
        const isMargin = market === 'margin';
        marginTypeWrap.style.display = isMargin ? '' : 'none';
        if (!isMargin) {
            appState.selectedMarginDataType = 'ohlcv';
            if (marginTypeSelect) marginTypeSelect.value = 'ohlcv';
        }
    }

    // 仅 swap/future 显示“衍生品数据类型”选择
    if (derivTypeWrap) {
        const isDeriv = (market === 'swap' || market === 'future');
        derivTypeWrap.style.display = isDeriv ? '' : 'none';
        if (!isDeriv) {
            appState.selectedDerivDataType = 'ohlcv';
            if (derivTypeSelect) derivTypeSelect.value = 'ohlcv';
        } else if (derivTypeSelect) {
            const swapOptions = [
                'ohlcv', 'ticker', 'orderbook', 'trades',
                'funding_rate', 'open_interest', 'mark_price', 'liquidations',
                'contract_info', 'market_info', 'snapshot'
            ];
            const futureOptions = [
                'ohlcv', 'ticker', 'orderbook', 'trades',
                'open_interest', 'basis', 'term_structure', 'settlement_price',
                'contract_info', 'market_info', 'snapshot'
            ];
            const opts = market === 'swap' ? swapOptions : futureOptions;
            updateSelectOptions(derivTypeSelect, opts, appState.selectedDerivDataType);
            appState.selectedDerivDataType = derivTypeSelect.value;
        }
    }
}

// ==================== 初始化 ====================

document.addEventListener('DOMContentLoaded', function () {
    console.log('应用初始化...');

    // 加载系统配置
    loadSystemConfig();

    // 初始化券商模式切换
    initBrokerModeToggle();

    // 初始化日期选择器
    initDateRangePicker();

    // 绑定事件
    bindEvents();

    // 初始化指标设置 UI
    applyAnalysisIndicatorStateToUI();

    // 根据 market 动态约束 exchange/timeframe，并禁用不支持的操作
    updateMarketUI();

    // 更新系统状态
    updateSystemStatus();

    // 每30秒更新一次系统状态
    setInterval(updateSystemStatus, 30000);

    // 实时更新系统时间
    updateSystemTime();
    setInterval(updateSystemTime, 1000);

    // Setup AlphaGen Metadata helpers
    setupAlphaGenMeta();

    // ML config & results
    loadMLConfig();
    loadMLSummary();

    // 初始化日志
    addLog('系统初始化完成', 'info');
});

// ==================== 配置加载 ====================

async function loadSystemConfig() {
    try {
        const data = await fetchJson('/api/config', {}, { timeoutMs: 8000, retries: 1 });
        appState.config = data;

        // 更新UI
        document.getElementById('supported-exchanges').textContent = data.exchanges.join(', ');
        document.getElementById('supported-markets').textContent = data.markets.join(', ');

        addLog('系统配置加载成功', 'success');
    } catch (error) {
        console.error('配置加载失败:', error);
        addLog('配置加载失败: ' + error.message, 'error');
    }
}

// ==================== 日期选择器 ====================

function initDateRangePicker() {
    const nowMs = Date.now();
    const startMs = nowMs - 30 * 24 * 60 * 60 * 1000;
    const startStr = formatUtc8DateTimeFromMs(startMs);
    const endStr = formatUtc8DateTimeFromMs(nowMs);

    const startInput = document.getElementById('startDateTime');
    const endInput = document.getElementById('endDateTime');
    if (startInput) startInput.value = startStr;
    if (endInput) endInput.value = endStr;

    // 开始时间选择器
    flatpickr('#startDateTime', {
        enableTime: true,
        dateFormat: 'Y-m-d H:i:S',
        defaultDate: startStr,
        time_24hr: true,
        locale: 'zh',
        minuteIncrement: 1,
        secondIncrement: 1,
        altInput: false,
        onClose: function (selectedDates) {
            if (selectedDates.length > 0) {
                updateDateRange();
            }
        }
    });

    // 结束时间选择器
    flatpickr('#endDateTime', {
        enableTime: true,
        dateFormat: 'Y-m-d H:i:S',
        defaultDate: endStr,
        time_24hr: true,
        locale: 'zh',
        minuteIncrement: 1,
        secondIncrement: 1,
        altInput: false,
        onClose: function (selectedDates) {
            if (selectedDates.length > 0) {
                updateDateRange();
            }
        }
    });

    // 初始化日期范围
    updateDateRange();
}

function updateDateRange() {
    const startInput = document.getElementById('startDateTime');
    const endInput = document.getElementById('endDateTime');
    const startMs = startInput && startInput.value ? parseUtc8DateTimeString(startInput.value) : NaN;
    const endMs = endInput && endInput.value ? parseUtc8DateTimeString(endInput.value) : NaN;

    if (startInput.value && endInput.value) {
        appState.dateRange = {
            start: startInput.value,
            end: endInput.value,
            startMs: Number.isFinite(startMs) ? startMs : null,
            endMs: Number.isFinite(endMs) ? endMs : null
        };
        addLog(`日期范围已更新: ${appState.dateRange.start} 到 ${appState.dateRange.end}`, 'info');
    }
}

function getDateRangeMs() {
    const dr = appState.dateRange || {};
    const startMs = Number.isFinite(dr.startMs) ? dr.startMs : (dr.start ? parseUtc8DateTimeString(dr.start) : NaN);
    const endMs = Number.isFinite(dr.endMs) ? dr.endMs : (dr.end ? parseUtc8DateTimeString(dr.end) : NaN);
    return { startMs, endMs };
}

function parseUtc8DateTimeString(value) {
    const s = String(value || '').trim();
    if (!s) return NaN;
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})[\sT](\d{2}):(\d{2}):(\d{2})$/);
    if (m) {
        const y = Number(m[1]);
        const mo = Number(m[2]) - 1;
        const d = Number(m[3]);
        const h = Number(m[4]);
        const mi = Number(m[5]);
        const sec = Number(m[6]);
        // Interpret input as UTC+8, convert to UTC ms
        return Date.UTC(y, mo, d, h - 8, mi, sec, 0);
    }
    const fallback = Date.parse(s);
    return Number.isNaN(fallback) ? NaN : fallback;
}

function formatUtc8DateTimeFromMs(utcMs) {
    const d = new Date(Number(utcMs) + 8 * 3600000);
    const yyyy = d.getUTCFullYear();
    const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(d.getUTCDate()).padStart(2, '0');
    const hh = String(d.getUTCHours()).padStart(2, '0');
    const mi = String(d.getUTCMinutes()).padStart(2, '0');
    const ss = String(d.getUTCSeconds()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}`;
}

function initBrokerModeToggle() {
    const toggle = document.getElementById('toggle-broker-mode');
    if (!toggle) return;

    const saved = localStorage.getItem('broker_mode');
    const enabled = saved === '1';
    toggle.checked = enabled;
    document.body.classList.toggle('broker-mode', enabled);

    toggle.addEventListener('change', () => {
        const on = !!toggle.checked;
        document.body.classList.toggle('broker-mode', on);
        localStorage.setItem('broker_mode', on ? '1' : '0');
    });
}

// ==================== 事件绑定 ====================

function bindEvents() {
    // 市场选择变化
    document.getElementById('exchange').addEventListener('change', function () {
        appState.selectedExchange = this.value;
        addLog(`交易所已切换为: ${this.value}`, 'info');

        // 切换交易所后清理交易对/结果，避免混用
        resetSymbolsUI();
        resetFetchedDataUI();
    });

    document.getElementById('market').addEventListener('change', function () {
        appState.selectedMarket = this.value;
        addLog(`市场已切换为: ${this.value}`, 'info');

        // 切换市场后清理交易对/结果，避免跨市场误用
        resetSymbolsUI();
        resetFetchedDataUI();

        updateMarketUI();
    });

    const timeframeSelect = document.getElementById('timeframe');
    if (timeframeSelect) {
        timeframeSelect.addEventListener('change', function () {
            appState.selectedTimeframe = this.value;
        });
    }

    const btnLoadTimeframes = document.getElementById('btn-load-timeframes');
    if (btnLoadTimeframes && timeframeSelect) {
        btnLoadTimeframes.addEventListener('click', async () => {
            try {
                btnLoadTimeframes.disabled = true;
                const prevText = btnLoadTimeframes.textContent;
                btnLoadTimeframes.textContent = '加载中...';
                const data = await fetchJson('/api/timeframes');
                if (data && Array.isArray(data.timeframes)) {
                    updateSelectOptions(timeframeSelect, data.timeframes, appState.selectedTimeframe);
                    appState.selectedTimeframe = timeframeSelect.value;
                } else {
                    showAlert('时间框架返回为空', 'warning');
                }
                btnLoadTimeframes.textContent = prevText || '加载时间框架';
            } catch (err) {
                console.error(err);
                showAlert('加载时间框架失败', 'danger');
                btnLoadTimeframes.textContent = '加载时间框架';
            } finally {
                btnLoadTimeframes.disabled = false;
            }
        });
    }

    const btnFetcherInfo = document.getElementById('btn-fetcher-info');
    const fetcherInfoResult = document.getElementById('fetcher-info-result');
    if (btnFetcherInfo) {
        btnFetcherInfo.addEventListener('click', async () => {
            try {
                btnFetcherInfo.disabled = true;
                const prevText = btnFetcherInfo.textContent;
                btnFetcherInfo.textContent = '加载中...';
                const payload = {
                    exchange: document.getElementById('exchange')?.value || appState.selectedExchange || 'binance',
                    market: document.getElementById('market')?.value || appState.selectedMarket || 'spot'
                };
                const data = await fetchJson('/api/fetcher/info', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (fetcherInfoResult) {
                    fetcherInfoResult.textContent = JSON.stringify(data || {}, null, 2);
                }
                btnFetcherInfo.textContent = prevText || 'Fetcher 信息';
            } catch (err) {
                console.error(err);
                showAlert('获取 Fetcher 信息失败', 'danger');
                btnFetcherInfo.textContent = 'Fetcher 信息';
            } finally {
                btnFetcherInfo.disabled = false;
            }
        });
    }

    const btnFetcherSnapshot = document.getElementById('btn-fetcher-snapshot');
    const snapshotSymbol = document.getElementById('snapshot-symbol');
    const snapshotTimeframe = document.getElementById('snapshot-timeframe');
    const snapshotInclude = document.getElementById('snapshot-include');
    const snapshotResult = document.getElementById('snapshot-result');
    if (btnFetcherSnapshot) {
        btnFetcherSnapshot.addEventListener('click', async () => {
            try {
                btnFetcherSnapshot.disabled = true;
                const prevText = btnFetcherSnapshot.textContent;
                btnFetcherSnapshot.textContent = '加载中...';
                const payload = {
                    exchange: document.getElementById('exchange')?.value || appState.selectedExchange || 'binance',
                    market: document.getElementById('market')?.value || appState.selectedMarket || 'spot',
                    symbol: snapshotSymbol?.value?.trim() || 'BTC/USDT',
                    timeframe: snapshotTimeframe?.value?.trim() || '1h',
                    include: snapshotInclude?.value?.trim() || ''
                };
                const data = await fetchJson('/api/fetcher/snapshot', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (snapshotResult) {
                    snapshotResult.textContent = JSON.stringify(data || {}, null, 2);
                }
                btnFetcherSnapshot.textContent = prevText || '获取快照';
            } catch (err) {
                console.error(err);
                showAlert('获取快照失败', 'danger');
                btnFetcherSnapshot.textContent = '获取快照';
            } finally {
                btnFetcherSnapshot.disabled = false;
            }
        });
    }

    const btnFetchRange = document.getElementById('btn-fetch-range');
    const rangeSymbol = document.getElementById('range-symbol');
    const rangeTimeframe = document.getElementById('range-timeframe');
    const rangeStart = document.getElementById('range-start');
    const rangeEnd = document.getElementById('range-end');
    const rangeResult = document.getElementById('range-result');
    if (btnFetchRange) {
        btnFetchRange.addEventListener('click', async () => {
            try {
                btnFetchRange.disabled = true;
                const prevText = btnFetchRange.textContent;
                btnFetchRange.textContent = '加载中...';
                const payload = {
                    exchange: document.getElementById('exchange')?.value || appState.selectedExchange || 'binance',
                    market: document.getElementById('market')?.value || appState.selectedMarket || 'spot',
                    symbol: rangeSymbol?.value?.trim() || 'BTC/USDT',
                    timeframe: rangeTimeframe?.value?.trim() || '1h',
                    start_time: rangeStart?.value?.trim() || '',
                    end_time: rangeEnd?.value?.trim() || ''
                };
                const data = await fetchJson('/api/data/fetch-range', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (rangeResult) {
                    rangeResult.textContent = JSON.stringify(data || {}, null, 2);
                }
                btnFetchRange.textContent = prevText || '按区间获取';
            } catch (err) {
                console.error(err);
                showAlert('获取区间数据失败', 'danger');
                btnFetchRange.textContent = '按区间获取';
            } finally {
                btnFetchRange.disabled = false;
            }
        });
    }

    const spotTypeSelect = document.getElementById('spot-data-type');
    if (spotTypeSelect) {
        spotTypeSelect.addEventListener('change', function () {
            appState.selectedSpotDataType = this.value || 'ohlcv';
            addLog(`现货数据类型已切换为: ${appState.selectedSpotDataType}`, 'info');
            resetFetchedDataUI();
        });
    }

    const optionTypeSelect = document.getElementById('option-data-type');
    if (optionTypeSelect) {
        optionTypeSelect.addEventListener('change', function () {
            appState.selectedOptionDataType = this.value || 'ohlcv';
            addLog(`期权数据类型已切换为: ${appState.selectedOptionDataType}`, 'info');
            resetFetchedDataUI();
        });
    }

    const marginTypeSelect = document.getElementById('margin-data-type');
    if (marginTypeSelect) {
        marginTypeSelect.addEventListener('change', function () {
            appState.selectedMarginDataType = this.value || 'ohlcv';
            addLog(`杠杆数据类型已切换为: ${appState.selectedMarginDataType}`, 'info');
            resetFetchedDataUI();
        });
    }

    const derivTypeSelect = document.getElementById('deriv-data-type');
    if (derivTypeSelect) {
        derivTypeSelect.addEventListener('change', function () {
            appState.selectedDerivDataType = this.value || 'ohlcv';
            addLog(`衍生品数据类型已切换为: ${appState.selectedDerivDataType}`, 'info');
            resetFetchedDataUI();
        });
    }

    // 交易对搜索 - 改为加载后进行本地搜索
    document.getElementById('btn-load-symbols').addEventListener('click', function () {
        loadSymbols();
    });

    document.getElementById('symbol-search-input').addEventListener('input', function () {
        if (allSymbols.length === 0) {
            showAlert('请先点击"加载交易对"按钮', 'warning');
            return;
        }
        const filtered = filterSymbols(this.value);
        displaySymbolList(filtered);
    });

    // 获取数据按钮
    document.getElementById('btn-fetch-data').addEventListener('click', function () {
        fetchData();
    });

    // 保存数据按钮
    document.getElementById('btn-save-data').addEventListener('click', function () {
        saveData();
    });

    // ==================== AlphaGen / RL ====================
    const btnTrain = document.getElementById('btn-alphagen-train');
    if (btnTrain) {
        btnTrain.addEventListener('click', function () {
            alphagenStartTrain();
        });
    }

    const btnRefresh = document.getElementById('btn-alphagen-refresh');
    if (btnRefresh) {
        btnRefresh.addEventListener('click', function () {
            alphagenRefreshJob();
        });
    }

    const btnCancel = document.getElementById('btn-alphagen-cancel');
    if (btnCancel) {
        btnCancel.addEventListener('click', function () {
            alphagenCancelJob();
        });
    }

    const btnEval = document.getElementById('btn-alphagen-eval');
    if (btnEval) {
        btnEval.addEventListener('click', function () {
            alphagenEvaluateExpr();
        });
    }

    const btnExport = document.getElementById('btn-alphagen-export');
    if (btnExport) {
        btnExport.addEventListener('click', function () {
            alphagenStartExportTopk();
        });
    }

    const btnListExports = document.getElementById('btn-alphagen-list-exports');
    if (btnListExports) {
        btnListExports.addEventListener('click', function () {
            alphagenListExports();
        });
    }

    // AlphaGen 数据准备：加载本地 merged 文件列表
    // 兼容旧 id: btn-alphagen-load-merged
    const btnLoadMerged = document.getElementById('btn-alphagen-load-merged');
    const btnLoadFiles = document.getElementById('btn-alphagen-load-files');
    const _bindLoad = (btn) => {
        if (!btn) return;
        btn.addEventListener('click', function () {
            alphagenLoadMergedFiles();
        });
    };
    _bindLoad(btnLoadMerged);
    _bindLoad(btnLoadFiles);

    const btnInferSwap = document.getElementById('btn-alphagen-infer-swap');
    if (btnInferSwap) {
        btnInferSwap.addEventListener('click', function () {
            alphagenInferSwapAux();
        });
    }

    const btnBuildPanel = document.getElementById('btn-alphagen-build-panel');
    if (btnBuildPanel) {
        btnBuildPanel.addEventListener('click', function () {
            alphagenBuildPanel();
        });
    }

    const btnRecommendPanel = document.getElementById('btn-alphagen-recommend-panel');
    if (btnRecommendPanel) {
        btnRecommendPanel.addEventListener('click', function () {
            alphagenApplyRecommendedPanel();
        });
    }

    const mergedSelect = document.getElementById('alphagen-merged-select');
    if (mergedSelect) {
        mergedSelect.addEventListener('change', function () {
            const opt = mergedSelect.selectedOptions?.[0];
            alphagenRenderMergedStatsFromOption(opt);
        });
    }

    const btnCheckPanel = document.getElementById('btn-alphagen-check-panel');
    if (btnCheckPanel) {
        btnCheckPanel.addEventListener('click', function () {
            alphagenCheckPanel();
        });
    }

    // ==================== ML / Research ====================
    const btnMLRun = document.getElementById('btn-ml-run');
    if (btnMLRun) {
        btnMLRun.addEventListener('click', function () {
            runMLPipeline();
        });
    }

    const btnMLRefresh = document.getElementById('btn-ml-refresh');
    if (btnMLRefresh) {
        btnMLRefresh.addEventListener('click', function () {
            loadMLSummary();
        });
    }

    const btnMLCancel = document.getElementById('btn-ml-cancel');
    if (btnMLCancel) {
        btnMLCancel.addEventListener('click', function () {
            cancelMLTask();
        });
    }


    // ==================== AlphaGen / RL ====================

    function alphagenGetCommonConfig() {
        const panel = (document.getElementById('alphagen-panel')?.value || '').trim();
        const rewardMode = (document.getElementById('alphagen-reward-mode')?.value || 'ic').trim();
        const nFolds = Number(document.getElementById('alphagen-n-folds')?.value || 3);
        const embargoBars = Number(document.getElementById('alphagen-embargo-bars')?.value || 60);
        const useFoldMedianIc = !!document.getElementById('alphagen-use-fold-median-ic')?.checked;
        return {
            panel,
            reward_mode: rewardMode,
            n_folds: nFolds,
            embargo_bars: embargoBars,
            use_fold_median_ic: useFoldMedianIc
        };
    }

    function alphagenSetJobUI(taskId, status, logs) {
        const idEl = document.getElementById('alphagen-job-id');
        const stEl = document.getElementById('alphagen-job-status');
        const logsEl = document.getElementById('alphagen-job-logs');
        if (idEl) idEl.textContent = taskId || '-';
        if (stEl) stEl.textContent = status || '-';
        if (logsEl) {
            const lines = Array.isArray(logs) ? logs : [];
            logsEl.textContent = lines.join('\n');
            logsEl.scrollTop = logsEl.scrollHeight;
        }
    }

    function alphagenRenderPanelStats(info) {
        const el = document.getElementById('alphagen-panel-stats');
        if (!el) return;
        if (!info) {
            el.innerHTML = '';
            return;
        }

        const sizeMb = info.size_bytes ? (info.size_bytes / 1024 / 1024).toFixed(2) : '-';
        const cols = Array.isArray(info.columns) ? info.columns : [];
        const missing = Array.isArray(info.required_missing) ? info.required_missing : [];
        const ok = !!info.ok;
        const timeRange = info.time_range ? `${info.time_range.start || '-'} ~ ${info.time_range.end || '-'}` : '-';
        const rowCount = info.row_count ?? '-';
        const warnRows = info.rows_truncated ? '（行数扫描已截断）' : '';
        const cls = ok ? 'alert-success' : 'alert-warning';

        el.innerHTML = `
            <div class="alert ${cls} p-2 mb-0">
                <div><strong>列数:</strong> ${cols.length} | <strong>行数:</strong> ${rowCount} ${warnRows}</div>
                <div><strong>大小:</strong> ${sizeMb} MB | <strong>时间列:</strong> ${info.time_column || '-'} | <strong>范围:</strong> ${timeRange}</div>
                <div><strong>缺失必备列:</strong> ${missing.length ? missing.join(', ') : '无'}</div>
            </div>
        `;
    }

    function alphagenRenderMergedStatsFromOption(opt) {
        const el = document.getElementById('alphagen-merged-stats');
        if (!el) return;
        if (!opt || !opt.value) {
            el.textContent = '';
            return;
        }
        const sym = opt.dataset.symbol || '-';
        const tf = opt.dataset.timeframe || '-';
        const fmt = opt.dataset.format || '-';
        const count = opt.dataset.count || '-';
        const start = opt.dataset.startTime || '-';
        const end = opt.dataset.endTime || '-';
        const gap = opt.dataset.gapRate ? `${(Number(opt.dataset.gapRate) * 100).toFixed(2)}%` : '-';
        const rec = opt.dataset.panelRecommend || '';
        const recText = rec ? `推荐 panel: ${rec}` : '';
        el.textContent = `symbol=${sym} | tf=${tf} | fmt=${fmt} | bars=${count} | span=${start} ~ ${end} | gap=${gap}${recText ? ' | ' + recText : ''}`;
    }

    function alphagenApplyRecommendedPanel() {
        const sel = document.getElementById('alphagen-merged-select');
        const opt = sel?.selectedOptions?.[0];
        const rec = opt?.dataset?.panelRecommend || '';
        if (!rec) {
            showAlert('当前选择没有推荐 panel 路径', 'warning');
            return;
        }
        const panelEl = document.getElementById('alphagen-panel');
        if (panelEl) panelEl.value = rec;
        showAlert('已填入推荐 panel 路径', 'success');
    }

    function alphagenRenderEvalCards(data) {
        const wrap = document.getElementById('alphagen-eval-cards');
        const saveEl = document.getElementById('alphagen-eval-save');
        if (!wrap) return;
        wrap.innerHTML = '';

        if (saveEl) {
            const fname = data?.eval_file || '';
            const url = data?.download_url || '';
            saveEl.innerHTML = fname && url ? `评估结果已落盘：<a href="${url}" target="_blank">${fname}</a>` : '';
        }

        const bd = data?.reward_breakdown || {};
        const exprInfo = data?.expr_info || {};
        const reward = data?.reward;

        const cards = [];
        cards.push({
            title: 'Reward',
            body: `reward=${Number(reward ?? 0).toFixed(6)}<br/>base=${Number(bd.base ?? 0).toFixed(6)}<br/>penalty=${Number(bd.penalty ?? 0).toFixed(6)}`
        });
        cards.push({
            title: 'Expr Info',
            body: `max_window=${exprInfo.max_window ?? '-'}<br/>max_shift=${exprInfo.max_shift ?? '-'}<br/>warmup=${exprInfo.warmup_bars ?? '-'}<br/>functions=${(exprInfo.functions || []).length}`
        });
        cards.push({
            title: 'Base Metrics',
            body: `base_ic=${Number(bd.base_ic ?? 0).toFixed(6)}<br/>base_trade=${Number(bd.base_trade ?? 0).toFixed(6)}<br/>mode=${bd.mode ?? '-'}`
        });

        wrap.innerHTML = cards.map(c => `
            <div class="col-md-4">
                <div class="card border-0 shadow-sm">
                    <div class="card-body p-2">
                        <div class="small text-muted">${c.title}</div>
                        <div class="small">${c.body}</div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    function alphagenRenderMultiEvalCards(data) {
        const wrap = document.getElementById('alphagen-eval-cards');
        const saveEl = document.getElementById('alphagen-eval-save');
        if (!wrap) return;
        wrap.innerHTML = '';

        if (saveEl) {
            const fname = data?.eval_file || '';
            const url = data?.download_url || '';
            saveEl.innerHTML = fname && url ? `评估结果已落盘：<a href="${url}" target="_blank">${fname}</a>` : '';
        }

        const stats = data?.result?.stats || {};
        const cards = [];
        cards.push({
            title: 'Multi-Asset Summary',
            body: `assets=${stats.n_assets ?? '-'}<br/>mean_rank_ic=${Number(stats.mean_rank_ic ?? 0).toFixed(4)}<br/>ic_sharpe=${Number(stats.rank_ic_sharpe ?? 0).toFixed(3)}`
        });
        cards.push({
            title: 'Time Range',
            body: `${stats.start_date ?? '-'} ~ ${stats.end_date ?? '-'}`
        });
        cards.push({
            title: 'Correlation',
            body: `avg_pairwise_corr=${Number(stats.avg_pairwise_corr ?? 0).toFixed(4)}`
        });

        wrap.innerHTML = cards.map(c => `
            <div class="col-md-4">
                <div class="card border-0 shadow-sm">
                    <div class="card-body p-2">
                        <div class="small text-muted">${c.title}</div>
                        <div class="small">${c.body}</div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    async function alphagenFetchMultiCandidates(panelPath) {
        const wrap = document.getElementById('alphagen-multi-candidates');
        if (!wrap) return;
        wrap.innerHTML = '';
        if (!panelPath) return;
        try {
            const resp = await fetch('/api/alphagen/multi-candidates', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ panel: panelPath })
            });
            const data = await resp.json();
            if (!data.success) throw new Error(data.error || 'Failed to match candidates');
            const candidates = Array.isArray(data.candidates) ? data.candidates : [];
            const warning = data.warning;
            const pat = data.pattern || '-';
            const list = candidates.length
                ? `<ul class="small mb-0">${candidates.map(p => `<li>${p}</li>`).join('')}</ul>`
                : '<div class="text-muted small">暂无候选文件</div>';
            const warnHtml = warning ? `<div class="alert alert-warning py-1 px-2 mb-2 small">${warning}</div>` : '';
            wrap.innerHTML = `
                ${warnHtml}
                <div class="small text-muted">pattern: ${pat} | count=${candidates.length}</div>
                ${list}
            `;
        } catch (e) {
            wrap.innerHTML = `<div class="text-danger small">候选匹配失败: ${e?.message || String(e)}</div>`;
        }
    }

    function alphagenStopPolling() {
        if (appState.alphagenPollTimer) {
            clearInterval(appState.alphagenPollTimer);
            appState.alphagenPollTimer = null;
        }
    }

    function alphagenStartPolling(taskId) {
        alphagenStopPolling();
        if (!taskId) return;
        appState.alphagenJobId = taskId;
        alphagenSetJobUI(taskId, 'polling', []);
        appState.alphagenPollTimer = setInterval(async () => {
            await alphagenRefreshJob();
        }, 2000);
    }

    async function alphagenLoadMergedFiles() {
        const ex = (document.getElementById('alphagen-spot-exchange')?.value || 'binance').trim();
        const sel = document.getElementById('alphagen-merged-select');
        if (!sel) {
            console.warn('alphagen-merged-select not found, skipping auto-load');
            return;
        }
        try {
            showLoading();
            const resp = await fetch(`/api/alphagen/spot-merged-files?exchange=${encodeURIComponent(ex)}`);
            const data = await resp.json();
            if (!data.success) throw new Error(data.error || 'Failed to load merged files');
            const files = Array.isArray(data.files) ? data.files : [];
            sel.innerHTML = '';
            const opt0 = document.createElement('option');
            opt0.value = '';
            opt0.textContent = files.length > 0 ? `（共 ${files.length} 个，先选一个；不显示就点右侧 List）` : '（暂无数据：请先在“数据管理”里保存到 storage）';
            sel.appendChild(opt0);
            for (const f of files) {
                const opt = document.createElement('option');
                opt.value = f.path || '';
                if (f.symbol) opt.dataset.symbol = f.symbol;
                if (f.timeframe) opt.dataset.timeframe = f.timeframe;
                if (f.format) opt.dataset.format = f.format;
                if (f.count != null) opt.dataset.count = f.count;
                if (f.start_time) opt.dataset.startTime = f.start_time;
                if (f.end_time) opt.dataset.endTime = f.end_time;
                if (f.gap_rate != null) opt.dataset.gapRate = f.gap_rate;
                if (f.panel_recommend) opt.dataset.panelRecommend = f.panel_recommend;
                const sym = f.symbol || f.name;
                const tf = f.timeframe ? ` ${f.timeframe}` : '';
                const fmt = f.format ? ` [${f.format}]` : '';
                opt.textContent = `${sym}${tf}${fmt} — ${f.name}`;
                sel.appendChild(opt);
            }
            const optSel = sel?.selectedOptions?.[0];
            alphagenRenderMergedStatsFromOption(optSel);
            if (files.length > 0) {
                console.log(`Loaded ${files.length} merged files for AlphaGen`);
            } else {
                console.warn('No merged files found. Please save data first in Data Management tab.');
            }
        } catch (e) {
            console.error('Failed to load merged files:', e);
            const opt0 = document.createElement('option');
            opt0.value = '';
            opt0.textContent = '(加载失败，请点击 List 重试)';
            sel.innerHTML = '';
            sel.appendChild(opt0);
        } finally {
            hideLoading();
        }
    }

    async function alphagenInferSwapAux() {
        const ex = (document.getElementById('alphagen-spot-exchange')?.value || 'binance').trim();
        const mergedPath = (document.getElementById('alphagen-merged-select')?.value || '').trim();
        const fundingEl = document.getElementById('alphagen-swap-funding');
        const oiEl = document.getElementById('alphagen-swap-oi');
        if (!fundingEl || !oiEl) return;
        if (!mergedPath) {
            showAlert('请先在 Spot Merged Data 里选一个文件（json/parquet）', 'warning');
            return;
        }
        // Prefer symbol from option dataset (works for parquet paths too)
        const sel = document.getElementById('alphagen-merged-select');
        const opt = sel?.selectedOptions?.[0];
        let symbol = (opt?.dataset?.symbol || '').trim();
        // Fallback: extract symbol from filename: BTC_USDT_1m_merged.json -> BTC/USDT
        if (!symbol) {
            const fn = mergedPath.replace(/\\/g, '/').split('/').pop() || '';
            const core = fn.endsWith('_merged.json') ? fn.slice(0, -'_merged.json'.length) : fn;
            const toks = core.split('_');
            symbol = toks.length >= 2 ? `${toks[0]}/${toks[1]}` : '';
        }
        if (!symbol) {
            showAlert('无法解析 symbol（建议先点 List 重新加载列表，再选择一条带 symbol 的项）', 'warning');
            return;
        }
        try {
            showLoading();
            const resp = await fetch(`/api/alphagen/infer-swap-aux?exchange=${encodeURIComponent(ex)}&symbol=${encodeURIComponent(symbol)}`);
            const data = await resp.json();
            if (!data.success) throw new Error(data.error || 'Infer failed');
            const paths = data.paths || {};
            if (paths.funding) fundingEl.value = paths.funding;
            if (paths.open_interest) oiEl.value = paths.open_interest;
            showAlert('已自动匹配 swap funding/OI（若本地存在）', 'success');
        } catch (e) {
            console.error(e);
            showAlert('自动匹配失败: ' + (e?.message || String(e)), 'danger');
        } finally {
            hideLoading();
        }
    }

    async function alphagenBuildPanel() {
        const ex = (document.getElementById('alphagen-spot-exchange')?.value || 'binance').trim();
        const mergedPath = (document.getElementById('alphagen-merged-select')?.value || '').trim();
        const timeframe = (document.getElementById('alphagen-panel-timeframe')?.value || '1m').trim();
        const horizon = Number(document.getElementById('alphagen-panel-horizon')?.value || 60);
        const maxRows = Number(document.getElementById('alphagen-panel-max-rows')?.value || 200000);
        const swapFunding = (document.getElementById('alphagen-swap-funding')?.value || '').trim();
        const swapOi = (document.getElementById('alphagen-swap-oi')?.value || '').trim();

        if (!mergedPath) {
            showAlert('请先点 List 加载文件列表，然后选择一个文件（json/parquet）', 'warning');
            return;
        }

        const payload = {
            exchange: ex,
            merged_path: mergedPath,
            timeframe,
            horizon,
            max_rows: maxRows,
            swap_funding: swapFunding || null,
            swap_oi: swapOi || null,
        };

        try {
            showLoading();
            const resp = await fetch('/api/alphagen/build-panel', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            if (!data.success) throw new Error(data.error || 'Build panel failed');

            appState.alphagenJobId = data.task_id;
            alphagenStartPolling(data.task_id);
            showAlert(`生成 Panel 任务已提交: ${data.task_id}`, 'success');
            addLog(`AlphaGen build panel submitted: ${data.task_id}`, 'info');

            // Remember expected output path, fill it once job completes.
            appState.alphagenPendingPanelOut = data.out_csv;
        } catch (e) {
            console.error(e);
            showAlert('生成 Panel 失败: ' + (e?.message || String(e)), 'danger');
        } finally {
            hideLoading();
        }
    }

    async function alphagenCheckPanel() {
        const panel = (document.getElementById('alphagen-panel')?.value || '').trim();
        if (!panel) {
            showAlert('请填写 Panel CSV 路径', 'warning');
            return;
        }
        try {
            showLoading();
            const resp = await fetch('/api/alphagen/panel-info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ panel })
            });
            const data = await resp.json();
            if (!data.success) throw new Error(data.error || 'Panel check failed');
            alphagenRenderPanelStats(data.info);
            showAlert('Panel 预检完成', 'success');
        } catch (e) {
            console.error(e);
            alphagenRenderPanelStats(null);
            showAlert('Panel 预检失败: ' + (e?.message || String(e)), 'danger');
        } finally {
            hideLoading();
        }
    }

    async function alphagenCancelJob() {
        const taskId = appState.alphagenJobId;
        if (!taskId) {
            showAlert('当前没有 AlphaGen 任务可取消', 'warning');
            return;
        }
        try {
            showLoading();
            const resp = await fetch(`/api/alphagen/cancel/${taskId}`, { method: 'POST' });
            const data = await resp.json();
            if (!data.success) throw new Error(data.error || 'Cancel failed');
            const t = data.task_info || {};
            alphagenSetJobUI(t.task_id, t.status, t.logs);
            alphagenStopPolling();
            showAlert('任务已取消', 'success');
        } catch (e) {
            console.error(e);
            showAlert('取消任务失败: ' + (e?.message || String(e)), 'danger');
        } finally {
            hideLoading();
        }
    }

    async function alphagenRefreshJob() {
        const taskId = appState.alphagenJobId;
        if (!taskId) {
            showAlert('当前没有 AlphaGen 任务，请先启动训练或导出', 'warning');
            return;
        }
        try {
            const resp = await fetch(`/api/alphagen/job/${taskId}`);
            const data = await resp.json();
            if (!data.success) {
                throw new Error(data.error || 'Failed to query job');
            }
            const t = data.task_info || {};
            alphagenSetJobUI(t.task_id, t.status, t.logs);
            if (['completed', 'error', 'timeout', 'cancelled'].includes(t.status)) {
                alphagenStopPolling();
                if (t.status === 'completed') {
                    addLog(`AlphaGen 任务完成: ${t.kind || ''}`, 'success');
                    if (t.kind === 'build_panel') {
                        const panelOut = t?.params?.out || appState.alphagenPendingPanelOut;
                        if (panelOut) {
                            const panelEl = document.getElementById('alphagen-panel');
                            if (panelEl) panelEl.value = String(panelOut);
                            showAlert('Panel 已生成并已自动填入 Panel CSV', 'success');
                        }
                    }
                    if (t.kind === 'export_topk') {
                        await alphagenListExports();
                    }
                } else if (t.status === 'cancelled') {
                    addLog('AlphaGen 任务已取消', 'warning');
                } else if (t.status === 'timeout') {
                    addLog(`AlphaGen 任务超时: ${t.error || 'timeout'}`, 'warning');
                } else {
                    addLog(`AlphaGen 任务失败: ${t.error || 'unknown error'}`, 'error');
                }
            }
        } catch (e) {
            console.error(e);
            addLog('查询 AlphaGen 任务失败: ' + (e?.message || String(e)), 'error');
        }
    }

    async function alphagenStartTrain() {
        const common = alphagenGetCommonConfig();
        if (!common.panel) {
            showAlert('请填写 Panel CSV 路径', 'warning');
            return;
        }
        const payload = {
            ...common,
            device: (document.getElementById('alphagen-device')?.value || 'cuda').trim(),
            timesteps: Number(document.getElementById('alphagen-timesteps')?.value || 2000),
            max_depth: Number(document.getElementById('alphagen-max-depth')?.value || 3),
            panel_rows: Number(document.getElementById('alphagen-panel-rows')?.value || 8000),
            seed: Number(document.getElementById('alphagen-seed')?.value || 7),
            out: (document.getElementById('alphagen-out')?.value || '').trim() || null
        };

        try {
            showLoading();
            const resp = await fetch('/api/alphagen/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            if (!data.success) {
                throw new Error(data.error || 'Failed to start train');
            }
            appState.alphagenJobId = data.task_id;
            alphagenStartPolling(data.task_id);
            showAlert(`训练任务已提交: ${data.task_id}`, 'success');
            addLog(`AlphaGen 训练任务已提交: ${data.task_id}`, 'info');
        } catch (e) {
            console.error(e);
            showAlert('启动训练失败: ' + (e?.message || String(e)), 'danger');
        } finally {
            hideLoading();
        }
    }

    async function alphagenEvaluateExpr() {
        const common = alphagenGetCommonConfig();
        const expr = (document.getElementById('alphagen-expr')?.value || '').trim();
        const outEl = document.getElementById('alphagen-eval-output');
        if (outEl) outEl.textContent = '';
        const cardsEl = document.getElementById('alphagen-eval-cards');
        if (cardsEl) cardsEl.innerHTML = '';
        const saveEl = document.getElementById('alphagen-eval-save');
        if (saveEl) saveEl.innerHTML = '';

        // Add mode
        const evalModeEl = document.getElementById('alphagen-eval-mode');
        const mode = evalModeEl ? evalModeEl.value : 'single';
        common.mode = mode;

        // Validate
        if (!common.panel) {
            showAlert('请填写 Panel CSV 路径', 'warning');
            return;
        }
        if (!expr) {
            showAlert('请填写 DSL 表达式', 'warning');
            return;
        }

        // Multi mode warning
        if (mode === 'multi' && !common.panel.includes('*') && !common.panel.includes('alphagen_panel')) {
            // Maybe user didn't select a pattern, but if they selected a file, evaluate_multi_asset might handle it 
            // or we should prompt them. For now, just let backend try.
        }

        if (mode === 'multi') {
            await alphagenFetchMultiCandidates(common.panel);
        }

        try {
            showLoading();
            const resp = await fetch('/api/alphagen/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...common,
                    expr,
                    ic_freq: '1D',
                    ic_method: 'spearman'
                })
            });
            const data = await resp.json();

            // Handle Multi Result
            if (data.mode === 'multi' && data.success) {
                const res = data.result;
                const stats = res.stats || {};
                let text = `=== Multi-Asset Cross-Sectional Evaluation ===\n\n`;
                text += `Expression: ${expr}\n`;
                text += `Assets Processed: ${stats.n_assets}\n`;
                text += `Time Range: ${stats.start_date} ~ ${stats.end_date}\n`;
                text += `Avg Pairwise Correlation: ${parseFloat(stats.avg_pairwise_corr).toFixed(4)}\n`;
                text += `Mean Rank IC: ${stats.mean_rank_ic != null ? parseFloat(stats.mean_rank_ic).toFixed(4) : 'N/A'}\n`;
                text += `Rank IC Sharpe: ${stats.rank_ic_sharpe != null ? parseFloat(stats.rank_ic_sharpe).toFixed(3) : 'N/A'}\n\n`;
                text += `Latest Ranks (Top 5 / Bottom 5):\n`;

                const ranks = Object.entries(res.latest_ranks || {});
                // Sort desc
                ranks.sort((a, b) => b[1] - a[1]);

                const top5 = ranks.slice(0, 5);
                const bot5 = ranks.slice(-5).reverse();

                top5.forEach(([k, v]) => text += `  Top ${k}: ${(v * 100).toFixed(1)}%\n`);
                text += '  ...\n';
                bot5.forEach(([k, v]) => text += `  Bot ${k}: ${(v * 100).toFixed(1)}%\n`);

                if (outEl) outEl.textContent = text;
                alphagenRenderMultiEvalCards(data);

                addLog(`Multi-Asset Eval: Mean IC=${parseFloat(stats.mean_rank_ic).toFixed(4)}`, 'success');
                return; // Stop here, skip single logic
            }

            if (!data.success) {
                throw new Error(data.error || 'Eval failed');
            }

            const showBreakdown = !!document.getElementById('alphagen-show-breakdown')?.checked;
            if (outEl) {
                if (showBreakdown && data.reward_breakdown) {
                    outEl.textContent = formatAlphaGenBreakdown(data.reward_breakdown) + "\n\n" + JSON.stringify(data, null, 2);
                } else {
                    outEl.textContent = JSON.stringify(data, null, 2);
                }
            }
            alphagenRenderEvalCards(data);
            addLog(`AlphaGen 评估完成 reward=${Number(data.reward).toFixed(6)}`, 'success');
        } catch (e) {
            console.error(e);
            if (outEl) outEl.textContent = String(e?.message || e);
            showAlert('评估失败: ' + (e?.message || String(e)), 'danger');
        } finally {
            hideLoading();
        }
    }

    function formatAlphaGenBreakdown(bd) {
        try {
            const mode = bd?.mode ?? '-';
            const base = bd?.base;
            const penalty = bd?.penalty;
            const reward = bd?.reward;
            const baseIc = bd?.base_ic;
            const baseTrade = bd?.base_trade;
            const lines = [];
            lines.push(`=== Reward Breakdown (mode=${mode}) ===`);
            lines.push(`reward = base - penalty`);
            lines.push(`base: ${Number(base).toFixed(6)} (base_ic=${Number(baseIc).toFixed(6)}, base_trade=${Number(baseTrade).toFixed(6)})`);
            lines.push(`penalty: ${Number(penalty).toFixed(6)}`);
            lines.push(`reward: ${Number(reward).toFixed(6)}`);
            lines.push('');
            lines.push('Killers (按 |weight*raw| 从大到小):');
            const terms = Array.isArray(bd?.penalty_terms) ? bd.penalty_terms : [];
            for (const t of terms) {
                const name = t?.name ?? '-';
                const w = t?.weight;
                const raw = t?.raw;
                const weighted = t?.weighted;
                if (weighted === null || weighted === undefined || Number.isNaN(Number(weighted))) {
                    continue;
                }
                lines.push(`- ${name}: weighted=${Number(weighted).toFixed(6)} (w=${Number(w).toFixed(3)}, raw=${Number(raw).toFixed(6)})`);
            }
            return lines.join('\n');
        } catch (e) {
            return 'Failed to format breakdown: ' + (e?.message || String(e));
        }
    }

    async function alphagenStartExportTopk() {
        const common = alphagenGetCommonConfig();
        const model = (document.getElementById('alphagen-export-model')?.value || '').trim();
        if (!common.panel) {
            showAlert('请填写 Panel CSV 路径', 'warning');
            return;
        }
        if (!model) {
            showAlert('请填写模型 .zip 路径', 'warning');
            return;
        }
        const payload = {
            ...common,
            model,
            max_depth: Number(document.getElementById('alphagen-max-depth')?.value || 3),
            seed: Number(document.getElementById('alphagen-seed')?.value || 7),
            episodes: Number(document.getElementById('alphagen-export-episodes')?.value || 200),
            topk: Number(document.getElementById('alphagen-export-topk')?.value || 50),
            panel_rows: Number(document.getElementById('alphagen-export-panel-rows')?.value || 12000)
        };
        try {
            showLoading();
            const resp = await fetch('/api/alphagen/export-topk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            if (!data.success) {
                throw new Error(data.error || 'Export failed');
            }
            appState.alphagenJobId = data.task_id;
            alphagenStartPolling(data.task_id);
            showAlert(`导出任务已提交: ${data.task_id}`, 'success');
            addLog(`AlphaGen 导出任务已提交: ${data.task_id}`, 'info');
        } catch (e) {
            console.error(e);
            showAlert('启动导出失败: ' + (e?.message || String(e)), 'danger');
        } finally {
            hideLoading();
        }
    }

    async function alphagenListExports() {
        const wrap = document.getElementById('alphagen-export-files');
        if (wrap) wrap.innerHTML = '';
        try {
            const resp = await fetch('/api/alphagen/exports');
            const data = await resp.json();
            if (!data.success) {
                throw new Error(data.error || 'Failed to list exports');
            }
            const files = Array.isArray(data.files) ? data.files : [];
            if (!wrap) return;
            if (files.length === 0) {
                wrap.innerHTML = '<div class="text-muted">暂无导出文件</div>';
                return;
            }
            const sorted = files
                .slice()
                .sort((a, b) => (b.modified_at || '').localeCompare(a.modified_at || ''));

            const items = sorted
                .map(f => {
                    const url = f.download_url || '#';
                    const name = f.name || '';
                    const sizeKb = Math.round((Number(f.size || 0) / 1024) * 10) / 10;
                    const mt = f.modified_at || '';
                    const isPreviewable = name.toLowerCase().endsWith('.csv') || name.toLowerCase().endsWith('.json');
                    const previewBtn = isPreviewable
                        ? `<button class="btn btn-sm btn-outline-primary ms-2" type="button" data-alphagen-preview="1" data-fname="${name}">预览</button>`
                        : '';
                    return `<li class="mb-1"><a href="${url}" target="_blank">${name}</a> <span class="text-muted">(${sizeKb}KB, ${mt})</span>${previewBtn}</li>`;
                })
                .join('');
            wrap.innerHTML = `<ul style="margin: 0; padding-left: 1.25rem;">${items}</ul>`;

            // bind preview buttons
            wrap.querySelectorAll('button[data-alphagen-preview="1"]').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const name = btn.getAttribute('data-fname') || '';
                    await alphagenPreviewTopkFile(name);
                });
            });

            // auto preview latest CSV if exists, else latest JSON
            const latestCsv = sorted.find(f => (f.name || '').toLowerCase().endsWith('.csv'));
            const latestJson = sorted.find(f => (f.name || '').toLowerCase().endsWith('.json'));
            const pick = (latestCsv && latestCsv.name) ? latestCsv.name : (latestJson && latestJson.name) ? latestJson.name : null;
            if (pick) {
                await alphagenPreviewTopkFile(pick);
            }
        } catch (e) {
            console.error(e);
            if (wrap) wrap.innerHTML = `<div class="text-danger">列出导出文件失败: ${e?.message || String(e)}</div>`;
        }
    }

    async function alphagenPreviewTopkFile(fileName) {
        const fileEl = document.getElementById('alphagen-preview-file');
        const tbody = document.getElementById('alphagen-preview-tbody');
        if (fileEl) fileEl.textContent = fileName || '-';
        if (tbody) tbody.innerHTML = '';
        if (!fileName) {
            return;
        }
        try {
            const resp = await fetch(`/api/alphagen/preview?file=${encodeURIComponent(fileName)}&limit=20`);
            const data = await resp.json();
            if (!data.success) {
                throw new Error(data.error || 'Preview failed');
            }
            const rows = Array.isArray(data.rows) ? data.rows : [];
            if (!tbody) return;
            if (rows.length === 0) {
                tbody.innerHTML = '<tr><td colspan="2" class="text-muted">无可预览内容</td></tr>';
                return;
            }
            const html = rows.map(r => {
                const reward = r && (r.reward ?? r['reward']);
                const expr = r && (r.expr ?? r['expr']);
                const rewardNum = Number(reward);
                const rewardStr = Number.isFinite(rewardNum) ? rewardNum.toFixed(6) : (reward === null || reward === undefined ? '' : String(reward));
                const exprStr = (expr === null || expr === undefined) ? '' : String(expr);
                return `<tr><td>${rewardStr}</td><td style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 0.9rem;">${escapeHtml(exprStr)}</td></tr>`;
            }).join('');
            tbody.innerHTML = html;
        } catch (e) {
            console.error(e);
            if (tbody) {
                tbody.innerHTML = `<tr><td colspan="2" class="text-danger">预览失败: ${escapeHtml(e?.message || String(e))}</td></tr>`;
            }
        }
    }
    // 保存缓存到本地（下载 JSON）
    const btnSaveCacheLocal = document.getElementById('btn-save-cache-local');
    if (btnSaveCacheLocal) {
        btnSaveCacheLocal.addEventListener('click', function () {
            saveCacheToLocal();
        });
    }

    // 导入本地缓存（合并到当前缓存）
    const btnImportCacheLocal = document.getElementById('btn-import-cache-local');
    const inputImportCacheFiles = document.getElementById('input-import-cache-files');
    if (btnImportCacheLocal && inputImportCacheFiles) {
        btnImportCacheLocal.addEventListener('click', function () {
            inputImportCacheFiles.value = '';
            inputImportCacheFiles.click();
        });

        inputImportCacheFiles.addEventListener('change', async function () {
            const files = Array.from(this.files || []);
            if (!files.length) return;
            await importCacheFiles(files);
        });
    }

    // 切换标签页时：关闭分析悬停气泡（避免挡住其它页面按钮）
    try {
        const hideBubble = () => {
            const api = window.__analysisHoverBubbleApi;
            if (api && typeof api.hide === 'function') {
                api.hide();
                return;
            }
            const el = document.getElementById('analysis-hover-bubble');
            if (el) el.style.display = 'none';
        };

        document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(btn => {
            btn.addEventListener('shown.bs.tab', (e) => {
                const target = e && e.target ? e.target.getAttribute('data-bs-target') : null;
                if (target !== '#data-analysis') {
                    hideBubble();
                }
            });
        });
    } catch {
        // ignore
    }

    // K线可视化按钮
    const btnViz = document.getElementById('btn-visualize-kline');
    if (btnViz) {
        btnViz.addEventListener('click', function () {
            readAnalysisIndicatorStateFromUI();
            visualizeKlineFromFetched();
        });
    }

    // 指标设置面板
    const btnInd = document.getElementById('btn-indicator-settings');
    const panel = document.getElementById('indicator-settings-panel');
    if (btnInd && panel) {
        btnInd.addEventListener('click', function () {
            panel.style.display = (panel.style.display === 'none' || panel.style.display === '') ? 'flex' : 'none';
            applyAnalysisIndicatorStateToUI();
        });
    }

    const setActiveIntervalButton = (val) => {
        const buttons = document.querySelectorAll('[data-interval]');
        buttons.forEach(btn => {
            const bVal = btn.getAttribute('data-interval');
            if (bVal === val) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    };

    const quickBtns = document.querySelectorAll('[data-interval]');
    quickBtns.forEach(btn => {
        btn.addEventListener('click', function () {
            const val = this.getAttribute('data-interval');
            const intervalEl = document.getElementById('analysis-interval');
            if (intervalEl && val) intervalEl.value = val;
            setActiveIntervalButton(val);
            visualizeKlineFromFetched();
        });
    });

    const intervalSelect = document.getElementById('analysis-interval');
    if (intervalSelect) {
        intervalSelect.addEventListener('change', function () {
            setActiveIntervalButton(this.value || 'raw');
            if (analysisCharts && (analysisCharts.price || analysisCharts.volume || analysisCharts.rsi || analysisCharts.lw)) {
                visualizeKlineFromFetched();
            }
        });
    }

    const btnDepth = document.getElementById('btn-refresh-depth');
    if (btnDepth) {
        btnDepth.addEventListener('click', function () {
            const select = document.getElementById('analysis-symbol');
            const symbol = select ? select.value : '';
            if (!symbol) {
                showAlert('请先选择交易对再刷新深度图', 'warning');
                return;
            }
            renderDepthChartForSymbol(symbol);
        });
    }

    const strategyEl = document.getElementById('analysis-signal-strategy');
    if (strategyEl) {
        strategyEl.addEventListener('change', function () {
            if (analysisCharts && (analysisCharts.price || analysisCharts.volume || analysisCharts.rsi || analysisCharts.lw)) {
                visualizeKlineFromFetched();
            }
        });
    }

    // init active quick button
    const initInterval = document.getElementById('analysis-interval')?.value || 'raw';
    setActiveIntervalButton(initInterval);

    const indicatorIds = ['ind-sma20', 'ind-sma50', 'ind-boll', 'ind-rsi', 'ind-takerbuy', 'ind-ema12', 'ind-ema26', 'ind-vwap', 'ind-macd', 'ind-vwma20', 'ind-hl2', 'ind-signals', 'ind-kdj', 'ind-cci', 'ind-bbwidth', 'ind-atr'];
    indicatorIds.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('change', function () {
            readAnalysisIndicatorStateFromUI();
            // 已渲染时自动刷新，保持最简单交互
            if (analysisCharts && (analysisCharts.price || analysisCharts.volume || analysisCharts.rsi || analysisCharts.lw)) {
                visualizeKlineFromFetched();
            }
        });
    });

    // 清除选择按钮
    document.getElementById('btn-clear-select').addEventListener('click', function () {
        clearSelection();
    });

    // 管理模块事件
    // 这里会动态添加加载和搜索事件
    const btnLoadManageSymbols = document.getElementById('btn-load-manage-symbols');
    if (btnLoadManageSymbols) {
        btnLoadManageSymbols.addEventListener('click', function () {
            loadManageSymbols();
        });
    }

    const btnLoadLocalSymbols = document.getElementById('btn-load-local-symbols');
    if (btnLoadLocalSymbols) {
        btnLoadLocalSymbols.addEventListener('click', function () {
            loadManageSymbolsFromLocal();
        });
    }

    const btnLoadLocalData = document.getElementById('btn-load-local-data');
    if (btnLoadLocalData) {
        btnLoadLocalData.addEventListener('click', function () {
            loadAnalysisLocalData();
        });
    }

    const btnLoadLocalSelected = document.getElementById('btn-load-local-selected');
    if (btnLoadLocalSelected) {
        btnLoadLocalSelected.addEventListener('click', function () {
            loadAnalysisSelectedFromLocal();
        });
    }

    const btnLocalInc = document.getElementById('btn-local-incremental-fetch');
    if (btnLocalInc) {
        btnLocalInc.addEventListener('click', function () {
            incrementalFetchSelectedLocal();
        });
    }

    const btnLocalExport = document.getElementById('btn-local-export-standard');
    if (btnLocalExport) {
        btnLocalExport.addEventListener('click', function () {
            exportStandardizedSelectedLocal();
        });
    }

    const analysisMarketSelect = document.getElementById('analysis-market-select');
    if (analysisMarketSelect) {
        analysisMarketSelect.addEventListener('change', function () {
            document.getElementById('analysis-local-status').textContent = '尚未加载';
            document.getElementById('analysis-symbol').innerHTML = '<option value="">请先点击“从本地加载”</option>';
            // 切换市场后，清空已加载结果避免混用
            appState.fetchedData = {};
            displayFetchResults(appState.fetchedData);
        });
    }

    const analysisExchangeSelect = document.getElementById('analysis-exchange-select');
    if (analysisExchangeSelect) {
        analysisExchangeSelect.addEventListener('change', function () {
            document.getElementById('analysis-local-status').textContent = '尚未加载';
            document.getElementById('analysis-symbol').innerHTML = '<option value="">请先点击“从本地加载”</option>';
            appState.fetchedData = {};
            displayFetchResults(appState.fetchedData);
        });
    }

    const analysisSymbolSelect = document.getElementById('analysis-symbol');
    if (analysisSymbolSelect) {
        // 多选模式下不自动加载；使用“批量加载”按钮。
        analysisSymbolSelect.addEventListener('change', function () {
            const selected = getSelectedValues(this);
            const statusEl = document.getElementById('analysis-local-status');
            if (statusEl) {
                statusEl.textContent = selected.length ? `已选择 ${selected.length} 个` : '尚未选择';
            }
            const symbolLabel = selected.length === 1 ? selected[0] : (selected.length ? `${selected.length} 个` : '--');
            updateBrokerHeader(symbolLabel, NaN, NaN, NaN);
        });
    }

    const btnRefreshLocalIndex = document.getElementById('btn-refresh-local-index');
    if (btnRefreshLocalIndex) {
        btnRefreshLocalIndex.addEventListener('click', function () {
            refreshLocalIndexTable();
        });
    }

    const manageSymbolSearchInput = document.getElementById('manage-symbol-search');
    if (manageSymbolSearchInput) {
        manageSymbolSearchInput.addEventListener('input', function () {
            searchSymbolsForManage(this.value);
        });
    }

    document.getElementById('btn-save-to-storage').addEventListener('click', function () {
        saveToStorage();
    });

    document.getElementById('btn-export-data').addEventListener('click', function () {
        exportData();
    });

    const btnAutoUpdateToMidnight = document.getElementById('btn-auto-update-to-midnight');
    if (btnAutoUpdateToMidnight) {
        btnAutoUpdateToMidnight.addEventListener('click', function () {
            autoUpdateManageSymbolsToMidnight();
        });
    }

    document.getElementById('btn-clear-manage').addEventListener('click', function () {
        clearManageSelection();
    });

    const btnClassifySymbols = document.getElementById('btn-classify-symbols');
    if (btnClassifySymbols) {
        btnClassifySymbols.addEventListener('click', function () {
            classifyManageSymbols();
        });
    }

    // 系统监控
    document.getElementById('btn-refresh-status').addEventListener('click', function () {
        updateSystemStatus();
    });

    const btnBrokerRefresh = document.getElementById('btn-broker-refresh');
    if (btnBrokerRefresh) {
        btnBrokerRefresh.addEventListener('click', function () {
            visualizeKlineFromFetched();
        });
    }

    const btnBrokerLayout = document.getElementById('btn-broker-layout');
    if (btnBrokerLayout) {
        btnBrokerLayout.addEventListener('click', function () {
            showAlert('布局切换已预留，可按需扩展（如分屏/多图）。', 'info');
        });
    }

}

function getSelectedValues(selectEl) {
    if (!selectEl) return [];
    const opts = Array.from(selectEl.selectedOptions || []);
    return opts.map(o => String(o.value || '').trim()).filter(Boolean);
}

async function importCacheFiles(files) {
    const readFileText = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result || ''));
            reader.onerror = () => reject(reader.error || new Error('File read error'));
            reader.readAsText(file);
        });
    };

    const extractResults = (obj) => {
        // 支持几种常见结构：
        // 1) { saved_at, meta, results }
        // 2) { results, meta }
        // 3) { task_info: { results } }
        // 4) 直接就是 results 对象（symbol -> {status,data})
        if (!obj || typeof obj !== 'object') return null;
        if (obj.results && typeof obj.results === 'object') return obj.results;
        if (obj.task_info && obj.task_info.results && typeof obj.task_info.results === 'object') return obj.task_info.results;

        // heuristics: 看起来像 results
        const keys = Object.keys(obj);
        if (keys.length && typeof obj[keys[0]] === 'object' && obj[keys[0]] && ('status' in obj[keys[0]] || 'data' in obj[keys[0]])) {
            return obj;
        }
        return null;
    };

    let merged = (appState.fetchedData && typeof appState.fetchedData === 'object') ? appState.fetchedData : {};
    let importedCount = 0;

    try {
        for (const f of files) {
            const text = await readFileText(f);
            let obj = null;
            try {
                obj = JSON.parse(text);
            } catch {
                addLog(`导入失败（非JSON）: ${f.name}`, 'error');
                continue;
            }

            const res = extractResults(obj);
            if (!res) {
                addLog(`导入失败（不识别的结构）: ${f.name}`, 'error');
                continue;
            }

            merged = mergeFetchedResults(merged, res);
            importedCount++;
        }

        if (importedCount === 0) {
            showAlert('未导入任何有效缓存文件（请确认选择的是缓存JSON）', 'warning');
            return;
        }

        appState.fetchedData = merged;
        refreshAnalysisSymbolOptions();
        displayFetchResults(appState.fetchedData);
        showAlert(`✅ 已导入并合并 ${importedCount} 个缓存文件`, 'success');
        addLog(`已导入并合并 ${importedCount} 个缓存文件，可直接可视化`, 'success');
    } catch (e) {
        console.error('导入本地缓存失败:', e);
        showAlert('导入本地缓存失败: ' + (e && e.message ? e.message : String(e)), 'danger');
        addLog('导入本地缓存失败: ' + (e && e.message ? e.message : String(e)), 'error');
    }
}

function saveCacheToLocal() {
    try {
        const results = appState.fetchedData || {};
        if (!results || typeof results !== 'object' || Object.keys(results).length === 0) {
            showAlert('暂无缓存数据可保存（请先获取数据或从磁盘恢复）', 'warning');
            return;
        }

        const { startMs, endMs } = getDateRangeMs();

        // 记录一些 meta，便于之后追溯
        const market = appState.selectedMarket || 'spot';
        const exchange = appState.selectedExchange || 'binance';
        const timeframe = appState.selectedTimeframe || '1h';

        const dataType = (() => {
            const m = String(market).toLowerCase();
            if (m === 'spot') return appState.selectedSpotDataType || 'ohlcv';
            if (m === 'option') return appState.selectedOptionDataType || 'ohlcv';
            if (m === 'margin') return appState.selectedMarginDataType || 'ohlcv';
            if (m === 'swap' || m === 'future') return appState.selectedDerivDataType || 'ohlcv';
            return 'snapshot';
        })();

        const payload = {
            saved_at: new Date().toISOString(),
            meta: {
                market,
                exchange,
                timeframe,
                data_type: dataType,
                start_time: Number.isFinite(startMs) ? startMs : null,
                end_time: Number.isFinite(endMs) ? endMs : null
            },
            results
        };

        const jsonText = JSON.stringify(payload, null, 2);
        const blob = new Blob([jsonText], { type: 'application/json;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        const safe = (s) => String(s || '').replaceAll('/', '_').replaceAll('\\', '_').replaceAll(':', '_');
        link.setAttribute('download', `cache_${safe(exchange)}_${safe(market)}_${safe(timeframe)}_${Date.now()}.json`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        showAlert('✅ 缓存数据已保存到本地（JSON）', 'success');
        addLog('缓存数据已保存到本地（JSON）', 'success');
    } catch (e) {
        console.error('保存缓存到本地失败:', e);
        showAlert('保存缓存到本地失败: ' + (e && e.message ? e.message : String(e)), 'danger');
        addLog('保存缓存到本地失败: ' + (e && e.message ? e.message : String(e)), 'error');
    }
}

function refreshAnalysisSymbolOptions() {
    const select = document.getElementById('analysis-symbol');
    if (!select) return;

    const prev = select.value;
    const items = [];

    const results = appState.fetchedData || {};
    for (const [symbol, result] of Object.entries(results)) {
        if (result && result.status === 'success' && Array.isArray(result.data) && result.data.length > 0) {
            items.push(symbol);
        }
    }

    select.innerHTML = '';
    if (items.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '请先获取数据';
        select.appendChild(opt);
        return;
    }

    items.forEach(symbol => {
        const opt = document.createElement('option');
        opt.value = symbol;
        opt.textContent = symbol;
        select.appendChild(opt);
    });

    if (items.includes(prev)) {
        select.value = prev;
    } else {
        select.value = items[0];
    }
}

function visualizeKlineFromFetched() {
    const market = (appState.selectedMarket || 'spot').toLowerCase();
    const getActiveDataTypeForMarket = (mkt) => {
        const m = String(mkt || '').toLowerCase();
        if (m === 'spot') return String(appState.selectedSpotDataType || 'ohlcv').toLowerCase();
        if (m === 'option') return String(appState.selectedOptionDataType || 'ohlcv').toLowerCase();
        if (m === 'margin') return String(appState.selectedMarginDataType || 'ohlcv').toLowerCase();
        if (m === 'swap' || m === 'future') return String(appState.selectedDerivDataType || 'ohlcv').toLowerCase();
        // onchain/social 统一视为 snapshot
        if (m === 'onchain' || m === 'social') return 'snapshot';
        return 'ohlcv';
    };

    const dataType = getActiveDataTypeForMarket(market);
    const isOhlcvMode = !(market === 'onchain' || market === 'social') && (dataType === 'ohlcv');

    const select = document.getElementById('analysis-symbol');
    const symbol = select ? select.value : '';
    if (!symbol) {
        showAlert('请先在“数据获取”里获取数据，再选择交易对', 'warning');
        return;
    }

    const result = appState.fetchedData && appState.fetchedData[symbol];
    if (!result || result.status !== 'success' || !Array.isArray(result.data) || result.data.length === 0) {
        showAlert('该交易对暂无可视化数据', 'warning');
        return;
    }

    if (isOhlcvMode) {
        updateBrokerFromOhlcv(symbol, result.data);
    } else {
        updateBrokerHeader(symbol, NaN, NaN, NaN);
    }

    const intervalEl = document.getElementById('analysis-interval');
    const interval = intervalEl ? (intervalEl.value || 'raw') : 'raw';

    const setAnalysisModeHint = (text) => {
        const el = document.getElementById('analysis-mode-hint');
        if (el) el.textContent = text;
    };

    const hideIndicatorPanel = () => {
        const panel = document.getElementById('indicator-settings-panel');
        if (panel) panel.style.display = 'none';
    };

    const destroyCharts = () => {
        if (analysisCharts.price) {
            analysisCharts.price.destroy();
            analysisCharts.price = null;
        }
        if (analysisCharts.volume) {
            analysisCharts.volume.destroy();
            analysisCharts.volume = null;
        }
        if (analysisCharts.rsi) {
            analysisCharts.rsi.destroy();
            analysisCharts.rsi = null;
        }
    };

    const escapeHtml = (s) => {
        return String(s)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    };

    const formatScalar = (v) => {
        if (v == null) return '';
        if (typeof v === 'number') {
            if (!Number.isFinite(v)) return '';
            if (Number.isInteger(v) || Math.abs(v) >= 1e6) return String(v);
            return v.toFixed(6);
        }
        if (typeof v === 'boolean') return v ? 'true' : 'false';
        if (typeof v === 'string') return v;
        return '';
    };

    const renderSnapshotPreview = async (sym, rows, mkt, dt) => {
        const priceContainer = document.getElementById('price-chart');
        const volumeContainer = document.getElementById('volume-chart');
        const rsiContainer = document.getElementById('rsi-chart');
        const macdContainer = document.getElementById('macd-chart');
        const oscContainer = document.getElementById('osc-chart');
        const volatilityContainer = document.getElementById('volatility-chart');
        if (!priceContainer || !volumeContainer || !rsiContainer) return;

        destroyCharts();
        hideIndicatorPanel();
        if (intervalEl) intervalEl.value = 'raw';

        setAnalysisModeHint(`当前为非 OHLCV：market=${mkt}, data_type=${dt}。将展示快照/表格预览（后续如需要可扩展为时间序列图）。`);

        const safeRows = Array.isArray(rows) ? rows : [];
        if (safeRows.length === 0) {
            priceContainer.innerHTML = '<p class="text-muted text-center">该交易对暂无可视化数据</p>';
            volumeContainer.innerHTML = '<p class="text-muted text-center">非 OHLCV 模式</p>';
            rsiContainer.innerHTML = '<p class="text-muted text-center">非 OHLCV 模式</p>';
            return;
        }

        const getMs = (rec) => {
            if (!rec || typeof rec !== 'object') return null;
            if (rec.timestamp != null) {
                const n = Number(rec.timestamp);
                return Number.isFinite(n) ? n : null;
            }
            if (rec.datetime) {
                const ms = Date.parse(rec.datetime);
                return Number.isNaN(ms) ? null : ms;
            }
            return null;
        };

        const renderOrderbookDepth = (sym2, rows2) => {
            const snap = rows2[0];
            if (!snap || (!snap.bids && !snap.asks)) return false;

            const bids = Array.isArray(snap.bids) ? snap.bids : [];
            const asks = Array.isArray(snap.asks) ? snap.asks : [];
            if (bids.length < 2 && asks.length < 2) return false;

            // 处理深度图数据：Bids 降序累计，Asks 升序累计
            // Bids: [price, amount]. Sort DESC price.
            const sortedBids = [...bids].sort((a, b) => b[0] - a[0]);
            // Asks: [price, amount]. Sort ASC price.
            const sortedAsks = [...asks].sort((a, b) => a[0] - b[0]);

            const bidPoints = [];
            let bidAcc = 0;
            for (const [p, a] of sortedBids) {
                bidAcc += Number(a);
                bidPoints.push({ x: Number(p), y: bidAcc });
            }
            // Reverse bids for chart (low to high price) so it connects correctly? 
            // Actually Chart.js scatter/line sorts by X usually, but 'fill' logic matters.
            // Depth chart usually: X=Price, Y=Sum.
            // Bids area: from left (low price) to best bid? No, usually high to low.
            // Let's standard: X axis is Price.
            // Bids: Green. On left side (lower prices) up to Mid. 
            // Asks: Red. On right side (higher prices) from Mid.

            const askPoints = [];
            let askAcc = 0;
            for (const [p, a] of sortedAsks) {
                askAcc += Number(a);
                askPoints.push({ x: Number(p), y: askAcc });
            }

            destroyCharts();
            hideIndicatorPanel();
            if (intervalEl) intervalEl.value = 'raw';

            const jsonBtnId = `btn-copy-json-${Date.now()}`;
            priceContainer.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <div>
                        <strong>${escapeHtml(sym2)}</strong> 深度图 (Orderbook Depth)
                    </div>
                    <button class="btn btn-outline-secondary btn-sm" id="${jsonBtnId}" type="button">复制JSON</button>
                </div>
                <canvas id="analysis-metric-canvas" style="height: 380px;"></canvas>
            `;
            volumeContainer.innerHTML = '<p class="text-muted text-center">Depth Chart Mode</p>';
            rsiContainer.innerHTML = '';

            const canvas = document.getElementById('analysis-metric-canvas');
            if (!canvas) return true;

            analysisCharts.price = new Chart(canvas.getContext('2d'), {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Bids (Buy)',
                            data: bidPoints,
                            borderColor: '#10b981', // Green
                            backgroundColor: 'rgba(16, 185, 129, 0.2)',
                            fill: true,
                            stepped: true,
                            pointRadius: 0
                        },
                        {
                            label: 'Asks (Sell)',
                            data: askPoints,
                            borderColor: '#ef4444', // Red
                            backgroundColor: 'rgba(239, 68, 68, 0.2)',
                            fill: true,
                            stepped: true,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index',
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Price' }
                        },
                        y: {
                            title: { display: true, text: 'Cumulative Volume' }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    return `${context.dataset.label}: Price ${context.parsed.x}, Vol ${context.parsed.y.toFixed(4)}`;
                                }
                            }
                        }
                    }
                }
            });
            return true;
        };

        const renderTradesScatter = (sym2, rows2) => {
            const snap = rows2[0];
            let trades = [];

            // Backend now standardizes return as payload object with 'trades' array
            // Structure: rows2 = [{ trades: [...], ... }]
            if (snap && Array.isArray(snap.trades)) {
                trades = snap.trades;
            } else if (Array.isArray(rows2) && rows2.length > 0 && rows2[0].price) {
                // Fallback for legacy format or direct list return
                trades = rows2;
            }

            if (trades.length < 5) return false;

            const buyPoints = [];
            const sellPoints = [];

            // Format: { x: timestamp, y: price, r: radius }
            trades.forEach(t => {
                const ts = t.timestamp;
                const price = Number(t.price);
                const amt = Number(t.amount);
                if (!ts || !price) return;

                // Radius scaling (log scale roughly)
                const r = Math.min(10, Math.max(2, Math.log(amt + 1) * 2));

                if (t.side === 'buy') {
                    buyPoints.push({ x: ts, y: price, r, _amt: amt });
                } else {
                    sellPoints.push({ x: ts, y: price, r, _amt: amt });
                }
            });

            destroyCharts();
            hideIndicatorPanel();

            const jsonBtnId = `btn-copy-json-${Date.now()}`;
            priceContainer.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <div>
                        <strong>${escapeHtml(sym2)}</strong> 成交分布 (Trade Scatter)
                    </div>
                    <button class="btn btn-outline-secondary btn-sm" id="${jsonBtnId}" type="button">复制JSON</button>
                </div>
                <canvas id="analysis-metric-canvas" style="height: 380px;"></canvas>
            `;
            volumeContainer.innerHTML = '<p class="text-muted text-center">Scatter Mode</p>';
            rsiContainer.innerHTML = '';

            const canvas = document.getElementById('analysis-metric-canvas');
            if (!canvas) return true;

            analysisCharts.price = new Chart(canvas.getContext('2d'), {
                type: 'bubble',
                data: {
                    datasets: [
                        {
                            label: 'Buy',
                            data: buyPoints,
                            backgroundColor: 'rgba(16, 185, 129, 0.6)',
                            borderColor: '#10b981',
                        },
                        {
                            label: 'Sell',
                            data: sellPoints,
                            backgroundColor: 'rgba(239, 68, 68, 0.6)',
                            borderColor: '#ef4444',
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'minute',
                                displayFormats: { minute: 'HH:mm' }
                            },
                            title: { display: true, text: 'Time' }
                        },
                        y: {
                            title: { display: true, text: 'Price' }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    const raw = context.raw;
                                    const d = new Date(raw.x).toLocaleTimeString();
                                    return `${context.dataset.label}: ${d} @ ${raw.y} (Vol: ${raw._amt})`;
                                }
                            }
                        }
                    }
                }
            });
            return true;
        };

        const renderSeriesChart = (sym2, rows2, mkt2, dt2) => {
            const points = [];
            const points2 = [];

            const pickY = (rec) => {
                if (!rec || typeof rec !== 'object') return null;
                if (dt2 === 'funding_rate') return rec.funding_rate;
                if (dt2 === 'open_interest') {
                    // 优先 value（通常更直观），否则 fallback amount
                    return (rec.open_interest_value != null) ? rec.open_interest_value : rec.open_interest;
                }
                return null;
            };

            const pickY2 = (rec) => {
                if (!rec || typeof rec !== 'object') return null;
                if (dt2 === 'funding_rate') return rec.predicted_rate;
                if (dt2 === 'open_interest') {
                    // 作为第二条线展示 amount（如果 value 为主线）
                    if (rec.open_interest_value != null && rec.open_interest != null) return rec.open_interest;
                    return null;
                }
                return null;
            };

            for (const rec of rows2) {
                const ms = getMs(rec);
                if (ms == null) continue;
                const y = Number(pickY(rec));
                if (Number.isFinite(y)) points.push({ x: ms, y });
                const yb = Number(pickY2(rec));
                if (Number.isFinite(yb)) points2.push({ x: ms, y: yb });
            }

            points.sort((a, b) => a.x - b.x);
            points2.sort((a, b) => a.x - b.x);

            if (points.length < 2) return false;

            destroyCharts();
            hideIndicatorPanel();
            if (intervalEl) intervalEl.value = 'raw';

            const jsonBtnId = `btn-copy-json-${Date.now()}`;
            priceContainer.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <div>
                        <strong>${escapeHtml(sym2)}</strong>
                        <span class="text-muted ms-2">market=${escapeHtml(mkt2)} data_type=${escapeHtml(dt2)}</span>
                    </div>
                    <button class="btn btn-outline-secondary btn-sm" id="${jsonBtnId}" type="button">复制该交易对JSON</button>
                </div>
                <canvas id="analysis-metric-canvas" style="height: 380px;"></canvas>
            `;
            volumeContainer.innerHTML = '<p class="text-muted text-center">非 OHLCV 模式：成交量/指标图暂不展示</p>';
            rsiContainer.innerHTML = '<p class="text-muted text-center">非 OHLCV 模式：RSI/指标图暂不展示</p>';

            const canvas = document.getElementById('analysis-metric-canvas');
            if (!canvas) return true;

            const fmtX = (ms) => {
                const d = new Date(ms);
                return Number.isNaN(d.getTime()) ? '' : d.toLocaleString('zh-CN');
            };

            const yTitle = (dt2 === 'funding_rate') ? 'Funding Rate' : 'Open Interest';
            const ds1Label = (dt2 === 'funding_rate') ? 'funding_rate' : ((points2.length > 0) ? 'open_interest_value' : 'open_interest');
            const ds2Label = (dt2 === 'funding_rate') ? 'predicted_rate' : 'open_interest';

            const rootStyle = (() => {
                try {
                    return getComputedStyle(document.documentElement);
                } catch {
                    return null;
                }
            })();
            const primaryColor = (rootStyle ? rootStyle.getPropertyValue('--primary-color') : '').trim() || '#1e40af';
            const secondaryColor = (rootStyle ? rootStyle.getPropertyValue('--secondary-color') : '').trim() || '#0891b2';

            analysisCharts.price = new Chart(canvas.getContext('2d'), {
                type: 'line',
                data: {
                    labels: points.map(p => fmtX(p.x)),
                    datasets: [
                        {
                            label: ds1Label,
                            data: points.map(p => p.y),
                            borderColor: primaryColor,
                            backgroundColor: 'rgba(30, 64, 175, 0.10)',
                            pointRadius: 0,
                            tension: 0.2
                        },
                        ...(points2.length > 1 ? [{
                            label: ds2Label,
                            data: points2.map(p => p.y),
                            borderColor: secondaryColor,
                            backgroundColor: 'rgba(8, 145, 178, 0.10)',
                            pointRadius: 0,
                            tension: 0.2
                        }] : [])
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true },
                        title: { display: true, text: `${sym2} - ${yTitle}` }
                    },
                    scales: {
                        x: {
                            ticks: { maxTicksLimit: 8 },
                            grid: { display: false }
                        },
                        y: {
                            ticks: {
                                callback: (v) => {
                                    if (dt2 === 'funding_rate') return (Number(v) * 100).toFixed(4) + '%';
                                    return String(v);
                                }
                            }
                        }
                    }
                }
            });

            setAnalysisModeHint(`已绘制时间序列：market=${mkt2}, data_type=${dt2}（${points.length}点）`);

            try {
                const btn = document.getElementById(jsonBtnId);
                if (btn) {
                    btn.addEventListener('click', async () => {
                        const text = JSON.stringify(rows2, null, 2);
                        const ok = await copyToClipboard(text);
                        if (ok) showAlert('✅ 已复制该交易对JSON到剪贴板', 'success');
                        else showAlert('复制失败：浏览器未允许剪贴板访问', 'warning');
                    });
                }
            } catch {
                // ignore
            }

            return true;
        };

        // 仅对明确的“序列型指标”尝试折线图
        if ((dt === 'funding_rate' || dt === 'open_interest') && safeRows.length >= 2) {
            const ok = renderSeriesChart(sym, safeRows, mkt, dt);
            if (ok) return;
        }

        // 尝试渲染 Orderbook 深度图和 Trades 散点图
        if (dt === 'orderbook') {
            if (renderOrderbookDepth(symbol, result.data)) return;
        }
        if (dt === 'trades') {
            if (renderTradesScatter(symbol, result.data)) return;
        }

        // 优先展示最多 50 行（例如 trades），否则展示最后一行
        const maxRows = (dt === 'trades') ? 50 : Math.min(10, safeRows.length);
        const view = (dt === 'trades')
            ? safeRows.slice(-maxRows)
            : [safeRows[safeRows.length - 1]];

        // 只展示“标量”字段，避免把 raw/trades/orderbook 这种大对象塞进表格
        const pickKeys = (obj) => {
            if (!obj || typeof obj !== 'object') return [];
            const keys = Object.keys(obj);
            const deny = new Set(['raw', 'trades', 'orderbook', 'liquidations', 'contract_info', 'market_info', 'ticker', 'greeks', 'option_price']);
            return keys.filter(k => !deny.has(k) && (typeof obj[k] !== 'object' || obj[k] == null));
        };

        const cols = pickKeys(view[0]);
        const headCols = cols.slice(0, 8); // 限制列数，保持最小可用

        const makeRowHtml = (obj) => {
            const t = (obj && (obj.datetime || obj.timestamp)) ? formatScalar(obj.datetime || obj.timestamp) : '';
            const cells = headCols.map(k => escapeHtml(formatScalar(obj[k])));
            return `
                <tr>
                    ${cells.map(c => `<td>${c}</td>`).join('')}
                </tr>
            `;
        };

        const headerHtml = headCols.map(k => `<th>${escapeHtml(k)}</th>`).join('');
        const bodyHtml = view.map(makeRowHtml).join('');

        const jsonBtnId = `btn-copy-json-${Date.now()}`;
        const preId = `pre-json-${Date.now()}`;

        priceContainer.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <div>
                    <strong>${escapeHtml(sym)}</strong>
                    <span class="text-muted ms-2">market=${escapeHtml(mkt)} data_type=${escapeHtml(dt)}</span>
                </div>
                <button class="btn btn-outline-secondary btn-sm" id="${jsonBtnId}" type="button">复制该交易对JSON</button>
            </div>
            <div class="data-table">
                <table class="table">
                    <thead><tr>${headerHtml}</tr></thead>
                    <tbody>${bodyHtml}</tbody>
                </table>
            </div>
            <details class="mt-2">
                <summary class="text-muted">查看完整 JSON（展开）</summary>
                <pre id="${preId}" style="white-space: pre-wrap; word-break: break-word; margin-top: 0.5rem;">${escapeHtml(JSON.stringify(safeRows, null, 2))}</pre>
            </details>
        `;

        volumeContainer.innerHTML = '<p class="text-muted text-center">非 OHLCV 模式：成交量/指标图暂不展示</p>';
        rsiContainer.innerHTML = '<p class="text-muted text-center">非 OHLCV 模式：RSI/指标图暂不展示</p>';
        if (macdContainer) macdContainer.innerHTML = '<p class="text-muted text-center">非 OHLCV 模式：MACD/指标图暂不展示</p>';
        if (oscContainer) oscContainer.innerHTML = '<p class="text-muted text-center">非 OHLCV 模式：KDJ/CCI 暂不展示</p>';
        if (volatilityContainer) volatilityContainer.innerHTML = '<p class="text-muted text-center">非 OHLCV 模式：ATR/带宽 暂不展示</p>';

        try {
            const btn = document.getElementById(jsonBtnId);
            if (btn) {
                btn.addEventListener('click', async () => {
                    const text = JSON.stringify(safeRows, null, 2);
                    const ok = await copyToClipboard(text);
                    if (ok) showAlert('✅ 已复制该交易对JSON到剪贴板', 'success');
                    else showAlert('复制失败：浏览器未允许剪贴板访问', 'warning');
                });
            }
        } catch {
            // ignore
        }
    };

    if (!isOhlcvMode) {
        renderSnapshotPreview(symbol, result.data, market, dataType);
        addLog(`已渲染 ${symbol} 的快照/指标预览（${market} ${dataType}）`, 'success');
        return;
    }

    setAnalysisModeHint('提示：OHLCV 将绘制K线/成交量；非 OHLCV（永续/期货指标、期权 Greeks、杠杆借币利率、链上/社交快照等）将显示快照表格。');

    const ohlcv = result.data;
    const baseTf = appState.selectedTimeframe || 'raw';
    const { data: viewData, label } = prepareOhlcvForView(ohlcv, interval, baseTf);

    renderAnalysisCharts(symbol, viewData, label);
    if (document.getElementById('analysis-sync-depth')?.checked) {
        renderDepthChartForSymbol(symbol);
    }
    addLog(`已渲染 ${symbol} 的K线/成交量图表（${label}）`, 'success');
}

function parseIntervalToMs(v) {
    const s = String(v || '').trim().toLowerCase();
    if (!s) return null;
    if (s === 'raw') return null;

    const m = /^([0-9]+)\s*([a-z]+)$/.exec(s);
    if (!m) return null;
    const n = parseInt(m[1], 10);
    const u = m[2];
    if (!Number.isFinite(n) || n <= 0) return null;

    if (u === 'm' || u === 'min' || u === 'mins' || u === 'minute' || u === 'minutes') return n * 60 * 1000;
    if (u === 'h' || u === 'hr' || u === 'hrs' || u === 'hour' || u === 'hours') return n * 60 * 60 * 1000;
    if (u === 'd' || u === 'day' || u === 'days') return n * 24 * 60 * 60 * 1000;
    if (u === 'w' || u === 'week' || u === 'weeks') return n * 7 * 24 * 60 * 60 * 1000;
    if (u === 'mo' || u === 'mon' || u === 'month' || u === 'months' || u === 'mth') return n * 30 * 24 * 60 * 60 * 1000;

    return null;
}

function prepareOhlcvForView(rawOhlcv, viewInterval, baseTimeframe) {
    const baseMs = parseIntervalToMs(baseTimeframe);
    const viewMs = parseIntervalToMs(viewInterval);

    // raw：不聚合
    if (!viewMs) {
        return { data: rawOhlcv, label: baseTimeframe || 'raw' };
    }

    // 如果用户选了更细粒度，但基础数据本身更粗，就无法“变细”，直接提示并用原始
    if (baseMs && viewMs < baseMs) {
        showAlert(`展示周期(${viewInterval})小于已获取周期(${baseTimeframe})，无法细分；将按原始周期展示`, 'warning');
        return { data: rawOhlcv, label: baseTimeframe || 'raw' };
    }

    const resampled = resampleOhlcv(rawOhlcv, viewMs);
    if (!resampled || resampled.length === 0) {
        showAlert(`聚合到 ${viewInterval} 后无有效K线，已回退原始周期`, 'warning');
        return { data: rawOhlcv, label: baseTimeframe || 'raw' };
    }
    return { data: resampled, label: viewInterval };
}

function resampleOhlcv(ohlcv, bucketMs) {
    if (!Array.isArray(ohlcv) || ohlcv.length === 0) return [];
    if (!bucketMs || bucketMs <= 0) return ohlcv;

    const toMs = (c) => {
        if (c && c.timestamp != null) return normalizeTimestampMs(c.timestamp);
        if (c && c.datetime) {
            const ms = Date.parse(c.datetime);
            if (!Number.isNaN(ms)) return ms;
        }
        return null;
    };

    // 升序，保证 open/close 正确
    const sorted = ohlcv
        .map(c => ({ ...c, _ms: toMs(c) }))
        .filter(c => c._ms != null && Number.isFinite(c._ms))
        .sort((a, b) => a._ms - b._ms);

    const buckets = new Map();
    for (const c of sorted) {
        const b = Math.floor(c._ms / bucketMs) * bucketMs;
        const cur = buckets.get(b);
        const o = Number(c.open);
        const h = Number(c.high);
        const l = Number(c.low);
        const cl = Number(c.close);
        const v = Number(c.volume);
        const qv = Number(c.quote_volume);
        const trades = Number(c.trades);
        const tbb = Number(c.taker_buy_base_volume);
        const tbq = Number(c.taker_buy_quote_volume);

        if (![o, h, l, cl].every(Number.isFinite)) {
            continue;
        }

        if (!cur) {
            buckets.set(b, {
                timestamp: b,
                datetime: new Date(b).toISOString(),
                open: o,
                high: h,
                low: l,
                close: cl,
                volume: Number.isFinite(v) ? v : 0,
                quote_volume: Number.isFinite(qv) ? qv : 0,
                trades: Number.isFinite(trades) ? trades : 0,
                taker_buy_base_volume: Number.isFinite(tbb) ? tbb : 0,
                taker_buy_quote_volume: Number.isFinite(tbq) ? tbq : 0
            });
        } else {
            cur.high = Math.max(cur.high, h);
            cur.low = Math.min(cur.low, l);
            cur.close = cl;
            cur.volume = (Number.isFinite(cur.volume) ? cur.volume : 0) + (Number.isFinite(v) ? v : 0);
            cur.quote_volume = (Number.isFinite(cur.quote_volume) ? cur.quote_volume : 0) + (Number.isFinite(qv) ? qv : 0);
            cur.trades = (Number.isFinite(cur.trades) ? cur.trades : 0) + (Number.isFinite(trades) ? trades : 0);
            cur.taker_buy_base_volume = (Number.isFinite(cur.taker_buy_base_volume) ? cur.taker_buy_base_volume : 0) + (Number.isFinite(tbb) ? tbb : 0);
            cur.taker_buy_quote_volume = (Number.isFinite(cur.taker_buy_quote_volume) ? cur.taker_buy_quote_volume : 0) + (Number.isFinite(tbq) ? tbq : 0);
        }
    }

    const out = Array.from(buckets.values()).sort((a, b) => a.timestamp - b.timestamp);
    // 聚合后可重算 vwap
    out.forEach(c => {
        const vol = Number(c.volume);
        const qv = Number(c.quote_volume);
        if (Number.isFinite(vol) && vol > 0 && Number.isFinite(qv) && qv > 0) {
            c.vwap = qv / vol;
        }
    });

    // 聚合后在前端补算指标（保证切换展示周期也能看 RSI/SMA 等）
    computeIndicatorsForOhlcv(out);
    return out;
}

function computeIndicatorsForOhlcv(ohlcv) {
    if (!Array.isArray(ohlcv) || ohlcv.length === 0) return;

    const closes = ohlcv.map(c => Number(c.close));
    const highs = ohlcv.map(c => Number(c.high));
    const lows = ohlcv.map(c => Number(c.low));
    const volumes = ohlcv.map(c => Number(c.volume));
    const n = closes.length;

    const sma = (period) => {
        const out = new Array(n).fill(null);
        let sum = 0;
        for (let i = 0; i < n; i++) {
            const v = closes[i];
            if (!Number.isFinite(v)) {
                out[i] = null;
                continue;
            }
            sum += v;
            if (i >= period) sum -= closes[i - period];
            if (i >= period - 1) out[i] = sum / period;
        }
        return out;
    };

    const ema = (period) => {
        const out = new Array(n).fill(null);
        const k = 2 / (period + 1);
        let prev = null;
        for (let i = 0; i < n; i++) {
            const v = closes[i];
            if (!Number.isFinite(v)) {
                out[i] = null;
                continue;
            }
            if (prev == null) {
                prev = v;
            } else {
                prev = v * k + prev * (1 - k);
            }
            out[i] = prev;
        }
        return out;
    };

    const rsi14 = (() => {
        const out = new Array(n).fill(null);
        const period = 14;
        let gains = 0;
        let losses = 0;
        for (let i = 1; i < n; i++) {
            const d = closes[i] - closes[i - 1];
            const g = d > 0 ? d : 0;
            const l = d < 0 ? -d : 0;

            if (i <= period) {
                gains += g;
                losses += l;
                if (i === period) {
                    const rs = losses === 0 ? Infinity : gains / losses;
                    out[i] = 100 - (100 / (1 + rs));
                }
                continue;
            }

            // Wilder smoothing
            gains = (gains * (period - 1) + g) / period;
            losses = (losses * (period - 1) + l) / period;
            const rs = losses === 0 ? Infinity : gains / losses;
            out[i] = 100 - (100 / (1 + rs));
        }
        return out;
    })();

    const sma20 = sma(20);
    const sma50 = sma(50);
    const ema12 = ema(12);
    const ema26 = ema(26);
    const macd = new Array(n).fill(null);
    for (let i = 0; i < n; i++) {
        if (ema12[i] != null && ema26[i] != null) macd[i] = ema12[i] - ema26[i];
    }
    // signal(9) on macd
    const macdSignal = (() => {
        const out = new Array(n).fill(null);
        const k = 2 / (9 + 1);
        let prev = null;
        for (let i = 0; i < n; i++) {
            const v = macd[i];
            if (!Number.isFinite(v)) {
                out[i] = null;
                continue;
            }
            if (prev == null) prev = v;
            else prev = v * k + prev * (1 - k);
            out[i] = prev;
        }
        return out;
    })();
    const macdHist = new Array(n).fill(null);
    for (let i = 0; i < n; i++) {
        if (macd[i] != null && macdSignal[i] != null) macdHist[i] = macd[i] - macdSignal[i];
    }

    // Bollinger(20,2)
    const bbMid = sma20;
    const bbUpper = new Array(n).fill(null);
    const bbLower = new Array(n).fill(null);
    for (let i = 19; i < n; i++) {
        const mid = bbMid[i];
        if (!Number.isFinite(mid)) continue;
        let mean = mid;
        let s = 0;
        for (let j = i - 19; j <= i; j++) {
            const v = closes[j];
            s += (v - mean) * (v - mean);
        }
        const std = Math.sqrt(s / 20);
        bbUpper[i] = mean + 2 * std;
        bbLower[i] = mean - 2 * std;
    }

    for (let i = 0; i < n; i++) {
        const c = ohlcv[i];
        if (!c) continue;
        if (Number.isFinite(rsi14[i])) c.rsi = rsi14[i];
        if (Number.isFinite(sma20[i])) c.sma_20 = sma20[i];
        if (Number.isFinite(sma50[i])) c.sma_50 = sma50[i];
        if (Number.isFinite(ema12[i])) c.ema_12 = ema12[i];
        if (Number.isFinite(ema26[i])) c.ema_26 = ema26[i];
        if (Number.isFinite(macd[i])) c.macd = macd[i];
        if (Number.isFinite(macdSignal[i])) c.macd_signal = macdSignal[i];
        if (Number.isFinite(macdHist[i])) c.macd_histogram = macdHist[i];
        if (Number.isFinite(bbUpper[i])) c.bollinger_upper = bbUpper[i];
        if (Number.isFinite(bbMid[i])) c.bollinger_middle = bbMid[i];
        if (Number.isFinite(bbLower[i])) c.bollinger_lower = bbLower[i];
    }

    // VWAP (if missing) + VWMA20 + HL2
    let cumPV = 0;
    let cumV = 0;
    for (let i = 0; i < n; i++) {
        const c = ohlcv[i];
        if (!c) continue;
        const v = Number.isFinite(volumes[i]) ? volumes[i] : 0;
        const tp = Number.isFinite(highs[i]) && Number.isFinite(lows[i]) && Number.isFinite(closes[i])
            ? (highs[i] + lows[i] + closes[i]) / 3
            : closes[i];

        if (c.vwap == null || !Number.isFinite(Number(c.vwap))) {
            const qv = Number(c.quote_volume);
            if (Number.isFinite(qv) && Number.isFinite(v) && v > 0) {
                c.vwap = qv / v;
            } else if (Number.isFinite(tp)) {
                c.vwap = tp;
            }
        }

        if (Number.isFinite(tp) && Number.isFinite(v)) {
            cumPV += tp * v;
            cumV += v;
            c.vwap_cum = (cumV > 0) ? (cumPV / cumV) : null;
        }

        // VWMA20
        if (i >= 19) {
            let sumPV = 0;
            let sumV = 0;
            for (let j = i - 19; j <= i; j++) {
                const vv = Number.isFinite(volumes[j]) ? volumes[j] : 0;
                const cc = Number.isFinite(closes[j]) ? closes[j] : null;
                if (cc != null) {
                    sumPV += cc * vv;
                    sumV += vv;
                }
            }
            c.vwma_20 = sumV > 0 ? (sumPV / sumV) : null;
        }

        if (Number.isFinite(highs[i]) && Number.isFinite(lows[i])) {
            c.hl2 = (highs[i] + lows[i]) / 2;
        }
    }

    // KDJ(9,3,3)
    const kPeriod = 9;
    let k = 50;
    let d = 50;
    for (let i = 0; i < n; i++) {
        if (i < kPeriod - 1) continue;
        let hh = -Infinity;
        let ll = Infinity;
        for (let j = i - kPeriod + 1; j <= i; j++) {
            if (Number.isFinite(highs[j])) hh = Math.max(hh, highs[j]);
            if (Number.isFinite(lows[j])) ll = Math.min(ll, lows[j]);
        }
        const c = ohlcv[i];
        if (!c || !Number.isFinite(closes[i]) || !Number.isFinite(hh) || !Number.isFinite(ll) || hh === ll) continue;
        const rsv = ((closes[i] - ll) / (hh - ll)) * 100;
        k = (2 / 3) * k + (1 / 3) * rsv;
        d = (2 / 3) * d + (1 / 3) * k;
        const jv = 3 * k - 2 * d;
        c.kdj_k = k;
        c.kdj_d = d;
        c.kdj_j = jv;
    }

    // CCI(20)
    const cciPeriod = 20;
    for (let i = 0; i < n; i++) {
        if (i < cciPeriod - 1) continue;
        let tpSum = 0;
        const tps = [];
        for (let j = i - cciPeriod + 1; j <= i; j++) {
            const tp = (highs[j] + lows[j] + closes[j]) / 3;
            tps.push(tp);
            tpSum += tp;
        }
        const ma = tpSum / cciPeriod;
        let md = 0;
        for (let j = 0; j < tps.length; j++) md += Math.abs(tps[j] - ma);
        md = md / cciPeriod;
        const c = ohlcv[i];
        if (!c || md === 0) continue;
        const tp = tps[tps.length - 1];
        c.cci = (tp - ma) / (0.015 * md);
    }

    // Bollinger Bandwidth
    for (let i = 0; i < n; i++) {
        const c = ohlcv[i];
        if (!c) continue;
        const bu = Number(c.bollinger_upper);
        const bl = Number(c.bollinger_lower);
        const bm = Number(c.bollinger_middle);
        if (Number.isFinite(bu) && Number.isFinite(bl) && Number.isFinite(bm) && bm !== 0) {
            c.bb_width = (bu - bl) / bm;
        }
    }

    // ATR(14)
    const atrPeriod = 14;
    let atr = null;
    for (let i = 1; i < n; i++) {
        const h = highs[i];
        const l = lows[i];
        const pc = closes[i - 1];
        if (!Number.isFinite(h) || !Number.isFinite(l) || !Number.isFinite(pc)) continue;
        const tr = Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));
        if (atr == null) {
            if (i >= atrPeriod) {
                let sum = 0;
                for (let j = i - atrPeriod + 1; j <= i; j++) {
                    const hh = highs[j];
                    const ll = lows[j];
                    const pcc = closes[j - 1];
                    const trj = Math.max(hh - ll, Math.abs(hh - pcc), Math.abs(ll - pcc));
                    sum += trj;
                }
                atr = sum / atrPeriod;
            }
        } else {
            atr = (atr * (atrPeriod - 1) + tr) / atrPeriod;
        }
        const c = ohlcv[i];
        if (c && atr != null) c.atr_14 = atr;
    }
}

function renderAnalysisChartsLightweight(symbol, ohlcvData, label) {
    const priceContainer = document.getElementById('price-chart');
    const volumeContainer = document.getElementById('volume-chart');
    const rsiContainer = document.getElementById('rsi-chart');
    const macdContainer = document.getElementById('macd-chart');
    const oscContainer = document.getElementById('osc-chart');
    const volatilityContainer = document.getElementById('volatility-chart');
    if (!priceContainer || !volumeContainer || !rsiContainer) return false;

    if (!(window.LightweightCharts && window.LightweightCharts.createChart)) return false;

    // 清理 Chart.js 图表
    if (analysisCharts.price) {
        analysisCharts.price.destroy();
        analysisCharts.price = null;
    }
    if (analysisCharts.volume) {
        analysisCharts.volume.destroy();
        analysisCharts.volume = null;
    }
    if (analysisCharts.rsi) {
        analysisCharts.rsi.destroy();
        analysisCharts.rsi = null;
    }

    // 清理轻量图表
    if (analysisCharts.lw) {
        try {
            analysisCharts.lw.charts.forEach(ch => ch.remove());
        } catch {
            // ignore
        }
        if (analysisCharts.lw.resizeObserver) {
            try {
                analysisCharts.lw.resizeObserver.disconnect();
            } catch {
                // ignore
            }
        }
        if (analysisCharts.lw.keydownHandler) {
            try {
                window.removeEventListener('keydown', analysisCharts.lw.keydownHandler);
            } catch {
                // ignore
            }
        }
        analysisCharts.lw = null;
    }

    // 准备容器
    priceContainer.innerHTML = '';
    volumeContainer.innerHTML = '';
    rsiContainer.innerHTML = '';
    priceContainer.style.position = 'relative';

    const maxPoints = 1500;
    const dataSlice = ohlcvData.length > maxPoints ? ohlcvData.slice(-maxPoints) : ohlcvData;

    const getUnixTime = (c, idx) => {
        if (c && c.timestamp != null && Number.isFinite(Number(c.timestamp))) {
            const ms = normalizeTimestampMs(c.timestamp);
            if (ms != null) return Math.floor(ms / 1000);
        }
        if (c && c.datetime) {
            const ms = Date.parse(c.datetime);
            if (!Number.isNaN(ms)) return Math.floor(ms / 1000);
        }
        return idx + 1;
    };

    const candleData = [];
    const volumeData = [];
    const sma20Data = [];
    const sma50Data = [];
    const bbUpperData = [];
    const bbLowerData = [];
    const rsiData = [];
    const ema12Data = [];
    const ema26Data = [];
    const vwapData = [];
    const vwma20Data = [];
    const hl2Data = [];
    const kData = [];
    const dData = [];
    const jData = [];
    const cciData = [];
    const bbWidthData = [];
    const atrData = [];
    const macdData = [];
    const macdSignalData = [];
    const macdHistData = [];

    for (let i = 0; i < dataSlice.length; i++) {
        const c = dataSlice[i];
        if (!c) continue;
        const time = getUnixTime(c, i);
        const o = Number(c.open);
        const h = Number(c.high);
        const l = Number(c.low);
        const cl = Number(c.close);
        const v = Number(c.volume);
        if ([o, h, l, cl].some(x => !Number.isFinite(x)) || time == null) continue;

        candleData.push({ time, open: o, high: h, low: l, close: cl });
        if (Number.isFinite(v)) {
            volumeData.push({
                time,
                value: v,
                color: cl >= o ? 'rgba(5,150,105,0.6)' : 'rgba(220,38,38,0.6)'
            });
        }

        const s20 = Number(c.sma_20);
        if (Number.isFinite(s20)) sma20Data.push({ time, value: s20 });
        const s50 = Number(c.sma_50);
        if (Number.isFinite(s50)) sma50Data.push({ time, value: s50 });
        const bu = Number(c.bollinger_upper);
        if (Number.isFinite(bu)) bbUpperData.push({ time, value: bu });
        const bl = Number(c.bollinger_lower);
        if (Number.isFinite(bl)) bbLowerData.push({ time, value: bl });
        const r = Number(c.rsi);
        if (Number.isFinite(r)) rsiData.push({ time, value: r });
        const e12 = Number(c.ema_12);
        if (Number.isFinite(e12)) ema12Data.push({ time, value: e12 });
        const e26 = Number(c.ema_26);
        if (Number.isFinite(e26)) ema26Data.push({ time, value: e26 });
        const vw = Number(c.vwap);
        if (Number.isFinite(vw)) vwapData.push({ time, value: vw });
        const vwma = Number(c.vwma_20);
        if (Number.isFinite(vwma)) vwma20Data.push({ time, value: vwma });
        const hl2 = Number(c.hl2);
        if (Number.isFinite(hl2)) hl2Data.push({ time, value: hl2 });
        const kdjK = Number(c.kdj_k);
        const kdjD = Number(c.kdj_d);
        const kdjJ = Number(c.kdj_j);
        if (Number.isFinite(kdjK)) kData.push({ time, value: kdjK });
        if (Number.isFinite(kdjD)) dData.push({ time, value: kdjD });
        if (Number.isFinite(kdjJ)) jData.push({ time, value: kdjJ });
        const cci = Number(c.cci);
        if (Number.isFinite(cci)) cciData.push({ time, value: cci });
        const bw = Number(c.bb_width);
        if (Number.isFinite(bw)) bbWidthData.push({ time, value: bw });
        const atr = Number(c.atr_14);
        if (Number.isFinite(atr)) atrData.push({ time, value: atr });
        const m = Number(c.macd);
        const ms = Number(c.macd_signal);
        const mh = Number(c.macd_histogram);
        if (Number.isFinite(m)) macdData.push({ time, value: m });
        if (Number.isFinite(ms)) macdSignalData.push({ time, value: ms });
        if (Number.isFinite(mh)) {
            macdHistData.push({
                time,
                value: mh,
                color: mh >= 0 ? 'rgba(5,150,105,0.6)' : 'rgba(220,38,38,0.6)'
            });
        }
    }

    if (candleData.length === 0) {
        return false;
    }

    const theme = getComputedStyle(document.documentElement);
    const primary = theme.getPropertyValue('--primary-color').trim() || '#1e40af';
    const success = theme.getPropertyValue('--success-color').trim() || '#059669';
    const danger = theme.getPropertyValue('--danger-color').trim() || '#dc2626';
    const dark = theme.getPropertyValue('--dark-color').trim() || '#1f2937';

    const ind = readAnalysisIndicatorStateFromUI();

    const chartTypeEl = document.getElementById('analysis-chart-type');
    const chartType = chartTypeEl ? chartTypeEl.value : 'candlestick';

    const createBaseOptions = (height) => ({
        width: priceContainer.clientWidth || 900,
        height,
        layout: {
            background: { color: '#ffffff' },
            textColor: '#111827'
        },
        grid: {
            vertLines: { color: '#f3f4f6' },
            horzLines: { color: '#f3f4f6' }
        },
        crosshair: {
            mode: 0
        },
        rightPriceScale: {
            borderVisible: false
        },
        timeScale: {
            borderVisible: false,
            timeVisible: true,
            secondsVisible: false
        },
        localization: {
            locale: 'zh-CN'
        }
    });

    const priceChart = LightweightCharts.createChart(priceContainer, createBaseOptions(priceContainer.clientHeight || 500));
    const volumeChart = LightweightCharts.createChart(volumeContainer, createBaseOptions(volumeContainer.clientHeight || 200));
    const rsiChart = LightweightCharts.createChart(rsiContainer, createBaseOptions(rsiContainer.clientHeight || 150));
    const macdChart = macdContainer ? LightweightCharts.createChart(macdContainer, createBaseOptions(macdContainer.clientHeight || 160)) : null;
    const oscChart = oscContainer ? LightweightCharts.createChart(oscContainer, createBaseOptions(oscContainer.clientHeight || 160)) : null;
    const volChart = volatilityContainer ? LightweightCharts.createChart(volatilityContainer, createBaseOptions(volatilityContainer.clientHeight || 140)) : null;

    // y-axis autoscale with margins
    try {
        priceChart.priceScale('right').applyOptions({ autoScale: true, scaleMargins: { top: 0.15, bottom: 0.1 } });
    } catch {
        // ignore
    }

    let candleSeries = null;
    let lineSeries = null;
    if (chartType === 'line') {
        lineSeries = priceChart.addAreaSeries({
            lineColor: primary,
            topColor: 'rgba(30,64,175,0.25)',
            bottomColor: 'rgba(30,64,175,0.02)',
            lineWidth: 2
        });
        const lineData = candleData.map(p => ({ time: p.time, value: p.close }));
        lineSeries.setData(lineData);
    } else {
        candleSeries = priceChart.addCandlestickSeries({
            upColor: success,
            downColor: danger,
            borderVisible: false,
            wickUpColor: success,
            wickDownColor: danger
        });
        candleSeries.setData(candleData);
    }

    const volumeSeries = volumeChart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'right',
        scaleMargins: { top: 0.2, bottom: 0 }
    });
    volumeSeries.setData(volumeData);

    const overlayOpts = { priceLineVisible: false, lastValueVisible: false };
    const sma20Series = priceChart.addLineSeries({ color: primary, lineWidth: 1, ...overlayOpts });
    const sma50Series = priceChart.addLineSeries({ color: dark, lineWidth: 1, ...overlayOpts });
    const bbUpperSeries = priceChart.addLineSeries({ color: 'rgba(2,132,199,0.7)', lineWidth: 1, ...overlayOpts });
    const bbLowerSeries = priceChart.addLineSeries({ color: 'rgba(2,132,199,0.7)', lineWidth: 1, ...overlayOpts });
    const ema12Series = priceChart.addLineSeries({ color: '#f59e0b', lineWidth: 1, ...overlayOpts });
    const ema26Series = priceChart.addLineSeries({ color: '#a855f7', lineWidth: 1, ...overlayOpts });
    const vwapSeries = priceChart.addLineSeries({ color: '#0ea5e9', lineWidth: 1, ...overlayOpts });
    const vwma20Series = priceChart.addLineSeries({ color: '#16a34a', lineWidth: 1, ...overlayOpts });
    const hl2Series = priceChart.addLineSeries({ color: '#64748b', lineWidth: 1, lineStyle: 2, ...overlayOpts });

    if (ind.sma20) sma20Series.setData(sma20Data); else sma20Series.applyOptions({ visible: false });
    if (ind.sma50) sma50Series.setData(sma50Data); else sma50Series.applyOptions({ visible: false });
    if (ind.boll) {
        bbUpperSeries.setData(bbUpperData);
        bbLowerSeries.setData(bbLowerData);
    } else {
        bbUpperSeries.applyOptions({ visible: false });
        bbLowerSeries.applyOptions({ visible: false });
    }
    if (ind.ema12) ema12Series.setData(ema12Data); else ema12Series.applyOptions({ visible: false });
    if (ind.ema26) ema26Series.setData(ema26Data); else ema26Series.applyOptions({ visible: false });
    if (ind.vwap) vwapSeries.setData(vwapData); else vwapSeries.applyOptions({ visible: false });
    if (ind.vwma20) vwma20Series.setData(vwma20Data); else vwma20Series.applyOptions({ visible: false });
    if (ind.hl2) hl2Series.setData(hl2Data); else hl2Series.applyOptions({ visible: false });

    const rsiSeries = rsiChart.addLineSeries({ color: primary, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    if (ind.rsi && rsiData.length > 0) rsiSeries.setData(rsiData); else rsiSeries.applyOptions({ visible: false });

    // RSI 70/30 参考线
    const rsiUpper = rsiChart.addLineSeries({ color: '#9ca3af', lineWidth: 1, lineStyle: 2 });
    const rsiLower = rsiChart.addLineSeries({ color: '#9ca3af', lineWidth: 1, lineStyle: 2 });
    if (ind.rsi && rsiData.length > 0) {
        rsiUpper.setData(rsiData.map(p => ({ time: p.time, value: 70 })));
        rsiLower.setData(rsiData.map(p => ({ time: p.time, value: 30 })));
    } else {
        rsiUpper.applyOptions({ visible: false });
        rsiLower.applyOptions({ visible: false });
    }

    if (rsiContainer) {
        rsiContainer.style.display = (ind.rsi && rsiData.length > 0) ? '' : 'none';
    }

    // KDJ / CCI
    if (oscChart) {
        const kLine = oscChart.addLineSeries({ color: '#10b981', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
        const dLine = oscChart.addLineSeries({ color: '#f59e0b', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
        const jLine = oscChart.addLineSeries({ color: '#3b82f6', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
        const cciLine = oscChart.addLineSeries({ color: '#8b5cf6', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });

        if (ind.kdj) {
            kLine.setData(kData);
            dLine.setData(dData);
            jLine.setData(jData);
        } else {
            kLine.applyOptions({ visible: false });
            dLine.applyOptions({ visible: false });
            jLine.applyOptions({ visible: false });
        }
        if (ind.cci) {
            cciLine.setData(cciData);
        } else {
            cciLine.applyOptions({ visible: false });
        }

        if (oscContainer) {
            const hasOsc = (ind.kdj && (kData.length || dData.length || jData.length)) || (ind.cci && cciData.length);
            oscContainer.style.display = hasOsc ? '' : 'none';
        }
    }

    // ATR / Bollinger Bandwidth
    if (volChart) {
        const atrLine = volChart.addLineSeries({ color: '#ef4444', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
        const bbwLine = volChart.addLineSeries({ color: '#0ea5e9', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });

        if (ind.atr) atrLine.setData(atrData); else atrLine.applyOptions({ visible: false });
        if (ind.bbwidth) bbwLine.setData(bbWidthData); else bbwLine.applyOptions({ visible: false });

        if (volatilityContainer) {
            const hasVol = (ind.atr && atrData.length) || (ind.bbwidth && bbWidthData.length);
            volatilityContainer.style.display = hasVol ? '' : 'none';
        }
    }

    // MACD 面板
    let macdLine = null;
    let macdSignal = null;
    let macdHist = null;
    if (macdChart) {
        macdLine = macdChart.addLineSeries({ color: '#2563eb', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
        macdSignal = macdChart.addLineSeries({ color: '#f59e0b', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
        macdHist = macdChart.addHistogramSeries({
            priceFormat: { type: 'price' },
            base: 0
        });
        if (ind.macd && (macdData.length > 0 || macdSignalData.length > 0 || macdHistData.length > 0)) {
            if (macdData.length > 0) macdLine.setData(macdData);
            if (macdSignalData.length > 0) macdSignal.setData(macdSignalData);
            if (macdHistData.length > 0) macdHist.setData(macdHistData);
            if (macdContainer) macdContainer.style.display = '';
        } else {
            macdLine.applyOptions({ visible: false });
            macdSignal.applyOptions({ visible: false });
            macdHist.applyOptions({ visible: false });
            if (macdContainer) macdContainer.style.display = 'none';
        }
    }

    // 时间范围联动
    let syncing = false;
    const updatePriceScaleToVisible = () => {
        try {
            const range = priceChart.timeScale().getVisibleRange();
            if (!range || candleData.length === 0) return;
            let lo = Infinity;
            let hi = -Infinity;
            for (const c of candleData) {
                if (!c || c.time < range.from || c.time > range.to) continue;
                if (Number.isFinite(c.low)) lo = Math.min(lo, c.low);
                if (Number.isFinite(c.high)) hi = Math.max(hi, c.high);
            }
            if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return;
            const pad = (hi - lo) * 0.05;
            const scale = priceChart.priceScale('right');
            if (scale && typeof scale.setVisibleRange === 'function') {
                scale.applyOptions({ autoScale: false });
                scale.setVisibleRange({ minValue: lo - pad, maxValue: hi + pad });
            }
        } catch {
            // ignore
        }
    };
    const syncFrom = (src) => {
        if (syncing) return;
        const range = src.timeScale().getVisibleRange();
        if (!range) return;
        syncing = true;
        try {
            [priceChart, volumeChart, rsiChart, macdChart, oscChart, volChart].forEach(ch => {
                if (ch !== src) ch.timeScale().setVisibleRange(range);
            });
            if (src === priceChart) updatePriceScaleToVisible();
        } finally {
            syncing = false;
        }
    };
    priceChart.timeScale().subscribeVisibleTimeRangeChange(() => syncFrom(priceChart));
    volumeChart.timeScale().subscribeVisibleTimeRangeChange(() => syncFrom(volumeChart));
    rsiChart.timeScale().subscribeVisibleTimeRangeChange(() => syncFrom(rsiChart));
    if (macdChart) macdChart.timeScale().subscribeVisibleTimeRangeChange(() => syncFrom(macdChart));
    if (oscChart) oscChart.timeScale().subscribeVisibleTimeRangeChange(() => syncFrom(oscChart));
    if (volChart) volChart.timeScale().subscribeVisibleTimeRangeChange(() => syncFrom(volChart));

    // 顶部信息栏
    const legend = document.createElement('div');
    legend.style.position = 'absolute';
    legend.style.top = '8px';
    legend.style.left = '12px';
    legend.style.padding = '6px 10px';
    legend.style.background = 'rgba(255,255,255,0.85)';
    legend.style.border = '1px solid #e5e7eb';
    legend.style.borderRadius = '6px';
    legend.style.fontSize = '12px';
    legend.style.color = '#111827';
    legend.style.pointerEvents = 'none';
    legend.textContent = `${symbol || ''} ${label || ''}`.trim();
    priceContainer.appendChild(legend);

    priceChart.subscribeCrosshairMove((param) => {
        if (!param || !param.time) return;
        const time = new Date(param.time * 1000).toLocaleString('zh-CN');
        if (candleSeries) {
            const candle = param.seriesData.get(candleSeries);
            if (!candle) return;
            legend.textContent = `${symbol || ''} ${label || ''}  ${time}  O:${candle.open} H:${candle.high} L:${candle.low} C:${candle.close}`.trim();
        } else if (lineSeries) {
            const pt = param.seriesData.get(lineSeries);
            if (!pt) return;
            legend.textContent = `${symbol || ''} ${label || ''}  ${time}  P:${pt.value}`.trim();
        }
    });

    // 交易信号标注 + 回测统计
    const stratEl = document.getElementById('analysis-signal-strategy');
    const strategy = stratEl ? stratEl.value : 'none';
    if (ind.signals && strategy && strategy !== 'none') {
        const markers = [];
        const trades = [];
        let position = null;
        for (let i = 0; i < dataSlice.length; i++) {
            const row = dataSlice[i];
            const time = getUnixTime(row, i);
            const price = Number(row.close);
            if (!Number.isFinite(price)) continue;

            let signal = 0;
            if (strategy === 'sma_cross') {
                const s20 = Number(row.sma_20);
                const s50 = Number(row.sma_50);
                if (Number.isFinite(s20) && Number.isFinite(s50)) {
                    signal = s20 > s50 ? 1 : s20 < s50 ? -1 : 0;
                }
            } else if (strategy === 'rsi_reversion') {
                const r = Number(row.rsi);
                if (Number.isFinite(r)) {
                    if (r < 30) signal = 1;
                    else if (r > 70) signal = -1;
                }
            } else if (strategy === 'kdj_cross') {
                const k = Number(row.kdj_k);
                const d = Number(row.kdj_d);
                if (Number.isFinite(k) && Number.isFinite(d)) {
                    signal = k > d ? 1 : k < d ? -1 : 0;
                }
            }

            if (!position && signal === 1) {
                position = { entryTime: time, entryPrice: price };
                markers.push({
                    time,
                    position: 'belowBar',
                    color: '#10b981',
                    shape: 'arrowUp',
                    text: 'BUY'
                });
            } else if (position && signal === -1) {
                const ret = (price - position.entryPrice) / position.entryPrice;
                trades.push({
                    ret,
                    entryTime: position.entryTime,
                    exitTime: time,
                    entryPrice: position.entryPrice,
                    exitPrice: price
                });
                markers.push({
                    time,
                    position: 'aboveBar',
                    color: '#ef4444',
                    shape: 'arrowDown',
                    text: 'SELL'
                });
                position = null;
            }
        }

        try {
            const targetSeries = candleSeries || lineSeries;
            if (targetSeries) targetSeries.setMarkers(markers);
        } catch {
            // ignore
        }

        // 简单回测统计
        const statsEl = document.getElementById('strategy-stats');
        if (statsEl) {
            if (trades.length === 0) {
                statsEl.style.display = '';
                const hint = document.getElementById('stat-strategy-hint');
                if (hint) hint.textContent = '暂无交易信号或回测样本不足。';
                const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
                setVal('stat-trades', '-');
                setVal('stat-winrate', '-');
                setVal('stat-avg', '-');
                setVal('stat-pnl', '-');
                setVal('stat-maxdd', '-');
                setVal('stat-pf', '-');
                setVal('stat-ann', '-');
                setVal('stat-sharpe', '-');
            } else {
                const returns = trades.map(t => t.ret);
                const wins = returns.filter(r => r > 0).length;
                const avg = returns.reduce((a, b) => a + b, 0) / returns.length;
                const wr = (wins / returns.length) * 100;

                // Equity curve
                let equity = 1;
                const equityCurve = [];
                for (const r of returns) {
                    equity *= (1 + r);
                    equityCurve.push(equity);
                }

                // Max drawdown
                let peak = -Infinity;
                let maxDd = 0;
                for (const v of equityCurve) {
                    if (v > peak) peak = v;
                    const dd = peak > 0 ? (v - peak) / peak : 0;
                    if (dd < maxDd) maxDd = dd;
                }

                // Profit factor
                const gains = returns.filter(r => r > 0).reduce((a, b) => a + b, 0);
                const losses = returns.filter(r => r < 0).reduce((a, b) => a + Math.abs(b), 0);
                const profitFactor = losses > 0 ? gains / losses : null;

                // Annualized return & Sharpe (approx by total duration)
                const firstTime = trades[0].entryTime;
                const lastTime = trades[trades.length - 1].exitTime;
                const totalSeconds = (Number(lastTime) - Number(firstTime)) / 1000;
                const annualSeconds = 365 * 24 * 3600;
                let annReturn = null;
                if (Number.isFinite(totalSeconds) && totalSeconds > 0) {
                    annReturn = Math.pow(equity, annualSeconds / totalSeconds) - 1;
                }

                // Sharpe from per-trade returns, annualized by trade frequency
                const mean = avg;
                const std = Math.sqrt(returns.reduce((s, r) => s + Math.pow(r - mean, 2), 0) / (returns.length || 1));
                let sharpe = null;
                if (std > 0 && Number.isFinite(totalSeconds) && totalSeconds > 0) {
                    const tradesPerYear = returns.length / (totalSeconds / annualSeconds);
                    sharpe = mean / std * Math.sqrt(tradesPerYear);
                }

                const totalPnlPct = (equity - 1) * 100;

                statsEl.style.display = '';
                const hint = document.getElementById('stat-strategy-hint');
                if (hint) hint.textContent = `策略：${strategy}`;
                const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
                setVal('stat-trades', String(returns.length));
                setVal('stat-winrate', `${wr.toFixed(1)}%`);
                setVal('stat-avg', `${(avg * 100).toFixed(2)}%`);
                setVal('stat-pnl', `${totalPnlPct.toFixed(2)}%`);
                setVal('stat-maxdd', `${(maxDd * 100).toFixed(2)}%`);
                setVal('stat-pf', profitFactor != null ? profitFactor.toFixed(2) : '-');
                setVal('stat-ann', annReturn != null ? `${(annReturn * 100).toFixed(2)}%` : '-');
                setVal('stat-sharpe', sharpe != null ? sharpe.toFixed(2) : '-');
            }
        }
    } else {
        const statsEl = document.getElementById('strategy-stats');
        if (statsEl) {
            statsEl.style.display = 'none';
        }
    }

    // 最后价格标注
    try {
        const last = candleData[candleData.length - 1];
        if (last && candleSeries) {
            candleSeries.createPriceLine({
                price: last.close,
                color: last.close >= last.open ? success : danger,
                lineWidth: 1,
                lineStyle: 0,
                axisLabelVisible: true,
                title: 'Last'
            });
        }
        if (last && lineSeries) {
            lineSeries.createPriceLine({
                price: last.close,
                color: primary,
                lineWidth: 1,
                lineStyle: 0,
                axisLabelVisible: true,
                title: 'Last'
            });
        }
    } catch {
        // ignore
    }

    priceChart.timeScale().fitContent();
    volumeChart.timeScale().fitContent();
    rsiChart.timeScale().fitContent();
    if (macdChart) macdChart.timeScale().fitContent();
    if (oscChart) oscChart.timeScale().fitContent();
    if (volChart) volChart.timeScale().fitContent();
    updatePriceScaleToVisible();

    // 键盘快捷键：+/- 缩放，0 复位，左右方向键平移
    const onKeyDown = (e) => {
        const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

        const scale = priceChart.timeScale();
        const range = scale.getVisibleRange();
        if (!range) return;

        const span = range.to - range.from;
        const zoom = (factor) => {
            const center = (range.from + range.to) / 2;
            const half = (span / factor) / 2;
            const newRange = { from: center - half, to: center + half };
            [priceChart, volumeChart, rsiChart, macdChart].forEach(ch => {
                if (ch) ch.timeScale().setVisibleRange(newRange);
            });
        };
        const pan = (dir) => {
            const shift = span * 0.1 * dir;
            const newRange = { from: range.from + shift, to: range.to + shift };
            [priceChart, volumeChart, rsiChart, macdChart].forEach(ch => {
                if (ch) ch.timeScale().setVisibleRange(newRange);
            });
        };

        if (e.key === '+' || e.key === '=') {
            zoom(1.25);
            e.preventDefault();
        } else if (e.key === '-' || e.key === '_') {
            zoom(0.8);
            e.preventDefault();
        } else if (e.key === '0') {
            [priceChart, volumeChart, rsiChart, macdChart].forEach(ch => {
                if (ch) ch.timeScale().fitContent();
            });
            e.preventDefault();
        } else if (e.key === 'ArrowLeft') {
            pan(-1);
            e.preventDefault();
        } else if (e.key === 'ArrowRight') {
            pan(1);
            e.preventDefault();
        }
    };
    window.addEventListener('keydown', onKeyDown);

    const resizeObserver = new ResizeObserver(() => {
        try {
            priceChart.applyOptions({ width: priceContainer.clientWidth, height: priceContainer.clientHeight || 500 });
            volumeChart.applyOptions({ width: volumeContainer.clientWidth, height: volumeContainer.clientHeight || 200 });
            rsiChart.applyOptions({ width: rsiContainer.clientWidth, height: rsiContainer.clientHeight || 150 });
            if (macdChart && macdContainer) macdChart.applyOptions({ width: macdContainer.clientWidth, height: macdContainer.clientHeight || 160 });
            if (oscChart && oscContainer) oscChart.applyOptions({ width: oscContainer.clientWidth, height: oscContainer.clientHeight || 160 });
            if (volChart && volatilityContainer) volChart.applyOptions({ width: volatilityContainer.clientWidth, height: volatilityContainer.clientHeight || 140 });
        } catch {
            // ignore
        }
    });
    resizeObserver.observe(priceContainer);
    resizeObserver.observe(volumeContainer);
    resizeObserver.observe(rsiContainer);
    if (macdContainer) resizeObserver.observe(macdContainer);
    if (oscContainer) resizeObserver.observe(oscContainer);
    if (volatilityContainer) resizeObserver.observe(volatilityContainer);

    // 鼠标框选缩放（Shift + 拖拽）
    const makeSelectionBox = () => {
        const box = document.createElement('div');
        box.style.position = 'absolute';
        box.style.top = '0';
        box.style.height = '100%';
        box.style.background = 'rgba(59,130,246,0.15)';
        box.style.border = '1px solid rgba(59,130,246,0.5)';
        box.style.display = 'none';
        box.style.pointerEvents = 'none';
        priceContainer.appendChild(box);
        return box;
    };
    const selectionBox = makeSelectionBox();
    let dragStartX = null;

    const onMouseDown = (e) => {
        if (!e.shiftKey) return;
        dragStartX = e.offsetX;
        selectionBox.style.left = `${dragStartX}px`;
        selectionBox.style.width = '0px';
        selectionBox.style.display = 'block';
    };
    const onMouseMove = (e) => {
        if (dragStartX == null) return;
        const x = e.offsetX;
        const left = Math.min(dragStartX, x);
        const width = Math.abs(x - dragStartX);
        selectionBox.style.left = `${left}px`;
        selectionBox.style.width = `${width}px`;
    };
    const onMouseUp = (e) => {
        if (dragStartX == null) return;
        const endX = e.offsetX;
        selectionBox.style.display = 'none';
        const left = Math.min(dragStartX, endX);
        const right = Math.max(dragStartX, endX);
        dragStartX = null;

        const from = priceChart.timeScale().coordinateToTime(left);
        const to = priceChart.timeScale().coordinateToTime(right);
        if (from && to) {
            const range = from < to ? { from, to } : { from: to, to: from };
            [priceChart, volumeChart, rsiChart, macdChart, oscChart, volChart].forEach(ch => {
                if (ch) ch.timeScale().setVisibleRange(range);
            });
        }
    };
    priceContainer.addEventListener('mousedown', onMouseDown);
    priceContainer.addEventListener('mousemove', onMouseMove);
    priceContainer.addEventListener('mouseup', onMouseUp);

    analysisCharts.lw = {
        charts: [priceChart, volumeChart, rsiChart, macdChart, oscChart, volChart].filter(Boolean),
        resizeObserver,
        keydownHandler: onKeyDown
    };

    return true;
}

function renderAnalysisCharts(symbol, ohlcvData, label) {
    if (renderAnalysisChartsLightweight(symbol, ohlcvData, label)) {
        return;
    }

    const priceContainer = document.getElementById('price-chart');
    const volumeContainer = document.getElementById('volume-chart');
    const rsiContainer = document.getElementById('rsi-chart');
    if (!priceContainer || !volumeContainer || !rsiContainer) return;

    // 准备容器
    priceContainer.innerHTML = '<canvas id="analysis-price-canvas"></canvas>';
    volumeContainer.innerHTML = '<canvas id="analysis-volume-canvas"></canvas>';
    rsiContainer.innerHTML = '<canvas id="analysis-rsi-canvas"></canvas>';

    // 销毁旧图表
    if (analysisCharts.price) {
        analysisCharts.price.destroy();
        analysisCharts.price = null;
    }
    if (analysisCharts.volume) {
        analysisCharts.volume.destroy();
        analysisCharts.volume = null;
    }
    if (analysisCharts.rsi) {
        analysisCharts.rsi.destroy();
        analysisCharts.rsi = null;
    }

    // 取最近 N 条，避免过多点导致卡顿
    const maxPoints = 800;
    const dataSlice = ohlcvData.length > maxPoints ? ohlcvData.slice(-maxPoints) : ohlcvData;

    // 悬停点位详情（默认提示）
    try {
        const detailEl = document.getElementById('analysis-point-details-text');
        if (detailEl) detailEl.textContent = '悬停在图表上查看该点的时间与指标详情；点击气泡可固定显示/隐藏。';
    } catch {
        // ignore
    }

    const getMs = (c) => {
        if (c && c.timestamp != null) return normalizeTimestampMs(c.timestamp);
        if (c && c.datetime) {
            const ms = Date.parse(c.datetime);
            if (!Number.isNaN(ms)) return ms;
        }
        return null;
    };

    const escapeHtml = (s) => {
        return String(s)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    };

    const fmtNum = (v, digits) => {
        const n = Number(v);
        if (!Number.isFinite(n)) return null;
        return (typeof digits === 'number') ? n.toFixed(digits) : String(n);
    };

    const fmtTime = (rec) => {
        const ms = getMs(rec);
        if (ms == null || !Number.isFinite(ms)) return null;
        const d = new Date(ms);
        if (Number.isNaN(d.getTime())) return null;
        return d.toLocaleString('zh-CN');
    };

    const formatPointDetailsHtml = (rec) => {
        if (!rec) return '';
        const lines = [];
        const t = fmtTime(rec);
        if (t) lines.push(`时间: ${t}`);

        const o = fmtNum(rec.open, 4);
        const h = fmtNum(rec.high, 4);
        const l = fmtNum(rec.low, 4);
        const c = fmtNum(rec.close, 4);
        const v = fmtNum(rec.volume, 4);
        if (o && h && l && c) lines.push(`OHLC: O=${o}, H=${h}, L=${l}, C=${c}`);
        if (v) lines.push(`volume: ${v}`);

        const qv = fmtNum(rec.quote_volume, 4);
        if (qv) lines.push(`quote_volume: ${qv}`);
        const trades = (rec.trades != null && Number.isFinite(Number(rec.trades))) ? String(Math.trunc(Number(rec.trades))) : null;
        if (trades) lines.push(`trades: ${trades}`);

        const tbb = fmtNum(rec.taker_buy_base_volume, 4);
        const tbq = fmtNum(rec.taker_buy_quote_volume, 4);
        if (tbb) lines.push(`taker_buy_base_volume: ${tbb}`);
        if (tbq) lines.push(`taker_buy_quote_volume: ${tbq}`);
        if (rec.taker_buy_base_volume != null && rec.volume != null) {
            const tbbN = Number(rec.taker_buy_base_volume);
            const volN = Number(rec.volume);
            if (Number.isFinite(tbbN) && Number.isFinite(volN) && volN > 0) {
                lines.push(`taker_buy% (base): ${(tbbN / volN * 100).toFixed(2)}%`);
            }
        }

        const vwap = fmtNum(rec.vwap, 4);
        if (vwap) lines.push(`vwap: ${vwap}`);

        const rsi = fmtNum(rec.rsi, 2);
        if (rsi) lines.push(`rsi(14): ${rsi}`);
        const sma20 = fmtNum(rec.sma_20, 4);
        const sma50 = fmtNum(rec.sma_50, 4);
        if (sma20) lines.push(`sma_20: ${sma20}`);
        if (sma50) lines.push(`sma_50: ${sma50}`);
        const ema12 = fmtNum(rec.ema_12, 4);
        const ema26 = fmtNum(rec.ema_26, 4);
        if (ema12) lines.push(`ema_12: ${ema12}`);
        if (ema26) lines.push(`ema_26: ${ema26}`);
        const macd = fmtNum(rec.macd, 4);
        const macdSignal = fmtNum(rec.macd_signal, 4);
        const macdHist = fmtNum(rec.macd_histogram, 4);
        if (macd) lines.push(`macd: ${macd}`);
        if (macdSignal) lines.push(`macd_signal: ${macdSignal}`);
        if (macdHist) lines.push(`macd_histogram: ${macdHist}`);
        const bu = fmtNum(rec.bollinger_upper, 4);
        const bm = fmtNum(rec.bollinger_middle, 4);
        const bl = fmtNum(rec.bollinger_lower, 4);
        if (bu) lines.push(`bollinger_upper: ${bu}`);
        if (bm) lines.push(`bollinger_middle: ${bm}`);
        if (bl) lines.push(`bollinger_lower: ${bl}`);

        return lines.length ? lines.map(s => escapeHtml(s)).join('<br>') : '该点暂无可展示的指标字段。';
    };

    const ensureHoverBubble = () => {
        let bubble = document.getElementById('analysis-hover-bubble');
        if (bubble) return bubble;

        bubble = document.createElement('div');
        bubble.id = 'analysis-hover-bubble';
        bubble.className = 'info-box';
        bubble.style.position = 'fixed';
        bubble.style.zIndex = '9999';
        bubble.style.display = 'none';
        bubble.style.alignItems = 'flex-start';
        bubble.style.maxWidth = '420px';
        bubble.style.cursor = 'pointer';
        bubble.innerHTML = `
            <i class="fas fa-info-circle"></i>
            <span class="info-box-text" id="analysis-hover-bubble-text"></span>
        `;
        document.body.appendChild(bubble);
        return bubble;
    };

    let bubblePinned = false;
    const showBubble = (rec, evt) => {
        const bubble = ensureHoverBubble();
        const textEl = document.getElementById('analysis-hover-bubble-text');
        if (textEl) textEl.innerHTML = formatPointDetailsHtml(rec);

        // Position near cursor
        const n = evt && evt.native ? evt.native : (evt && evt.event ? evt.event : evt);
        const cx = (n && typeof n.clientX === 'number') ? n.clientX : null;
        const cy = (n && typeof n.clientY === 'number') ? n.clientY : null;
        const x = (cx != null) ? cx : 20;
        const y = (cy != null) ? cy : 20;

        bubble.style.display = 'flex';
        bubble.style.left = (x + 12) + 'px';
        bubble.style.top = (y + 12) + 'px';

        // Clamp to viewport
        try {
            const r = bubble.getBoundingClientRect();
            let left = x + 12;
            let top = y + 12;
            if (r.right > window.innerWidth - 8) left = Math.max(8, x - r.width - 12);
            if (r.bottom > window.innerHeight - 8) top = Math.max(8, y - r.height - 12);
            bubble.style.left = left + 'px';
            bubble.style.top = top + 'px';
        } catch {
            // ignore
        }
    };

    const hideBubble = () => {
        const bubble = document.getElementById('analysis-hover-bubble');
        if (bubble) bubble.style.display = 'none';
    };

    // 暴露给全局（用于切换 tab 时强制关闭，避免挡住其它页面）
    try {
        window.__analysisHoverBubbleApi = {
            hide: () => {
                bubblePinned = false;
                hideBubble();
            },
            unpin: () => {
                bubblePinned = false;
            },
            isPinned: () => bubblePinned
        };
    } catch {
        // ignore
    }

    // 点击气泡：固定/隐藏
    try {
        const bubble = ensureHoverBubble();
        bubble.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (bubblePinned) {
                bubblePinned = false;
                hideBubble();
            } else {
                bubblePinned = true;
            }
        });
    } catch {
        // ignore
    }

    // 选中态（用于点击反馈/联动）
    let selectedMs = null;

    const msSeries = dataSlice
        .map(c => {
            const ms = getMs(c);
            return (ms != null && Number.isFinite(Number(ms))) ? Number(ms) : null;
        });

    const findNearestRecordByMs = (targetMs) => {
        const t = Number(targetMs);
        if (!Number.isFinite(t)) return null;

        // 收集有效点（msSeries 可能有 null）
        const pairs = [];
        for (let i = 0; i < msSeries.length; i++) {
            const ms = msSeries[i];
            if (ms != null) pairs.push([ms, i]);
        }
        if (pairs.length === 0) return null;

        // 二分
        let lo = 0;
        let hi = pairs.length - 1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            const ms = pairs[mid][0];
            if (ms === t) return dataSlice[pairs[mid][1]];
            if (ms < t) lo = mid + 1;
            else hi = mid - 1;
        }

        const right = Math.min(lo, pairs.length - 1);
        const left = Math.max(right - 1, 0);
        const dr = Math.abs(pairs[right][0] - t);
        const dl = Math.abs(pairs[left][0] - t);
        return dataSlice[(dr < dl) ? pairs[right][1] : pairs[left][1]];
    };

    const pickRecordFromClick = (chart, evt, activeEls) => {
        // 1) 优先用 activeEls
        if (activeEls && activeEls.length > 0) {
            const a = activeEls[0];

            // 对 bar 图：index 对应 dataSlice 的顺序
            if (chart && chart.config && chart.config.type === 'bar' && typeof a.index === 'number') {
                return dataSlice[a.index] || null;
            }

            const ds = chart && chart.data && chart.data.datasets ? chart.data.datasets[a.datasetIndex] : null;
            const pt = ds && Array.isArray(ds.data) ? ds.data[a.index] : null;
            const x = pt && (pt.x != null ? pt.x : null);
            if (x != null) return findNearestRecordByMs(x);
        }

        // 2) activeEls 为空：按鼠标位置算 x，再找最近时间点
        try {
            const xScale = chart && chart.scales ? chart.scales.x : null;
            if (xScale && typeof xScale.getValueForPixel === 'function') {
                const xPixel = (evt && typeof evt.x === 'number')
                    ? evt.x
                    : (evt && evt.native && typeof evt.native.offsetX === 'number')
                        ? evt.native.offsetX
                        : (evt && evt.native && typeof evt.native.layerX === 'number')
                            ? evt.native.layerX
                            : null;
                if (xPixel != null) {
                    const xVal = xScale.getValueForPixel(xPixel);
                    return findNearestRecordByMs(xVal);
                }
            }
        } catch {
            // ignore
        }

        return null;
    };

    const updateSelectionMarkers = () => {
        if (selectedMs == null) return;

        const updateForChart = (chart, yScaleId, datasetLabel) => {
            if (!chart) return;
            const yScale = chart.scales ? (chart.scales[yScaleId] || chart.scales.y) : null;
            if (!yScale) return;
            const yMin = yScale.min;
            const yMax = yScale.max;
            const ds = (chart.data && chart.data.datasets) ? chart.data.datasets.find(d => d && d.label === datasetLabel) : null;
            if (!ds) return;
            ds.data = [
                { x: selectedMs, y: yMin },
                { x: selectedMs, y: yMax }
            ];
            scheduleChartUpdate(chart, 'none');
        };

        updateForChart(analysisCharts.price, 'y', '__SELECTED__');
        updateForChart(analysisCharts.volume, 'y', '__SELECTED__');
        updateForChart(analysisCharts.rsi, 'y', '__SELECTED__');
    };

    // 蜡烛图数据：{ x(ms), o, h, l, c }
    const candleData = [];
    const volumeBarData = [];
    const quoteVolumeLineData = [];
    const tradesData = [];
    const sma20Data = [];
    const sma50Data = [];
    const bbUpperData = [];
    const bbLowerData = [];
    const rsiData = [];
    const takerBuyPctData = [];
    for (let i = 0; i < dataSlice.length; i++) {
        const c = dataSlice[i];
        const ms = getMs(c);
        const x = ms != null ? ms : i;
        candleData.push({
            x,
            o: Number(c.open),
            h: Number(c.high),
            l: Number(c.low),
            c: Number(c.close)
        });
        volumeBarData.push({ x, y: Number(c.volume) });
        quoteVolumeLineData.push({ x, y: Number(c.quote_volume) });
        tradesData.push(Number(c.trades));

        const sma20 = Number(c.sma_20);
        const sma50 = Number(c.sma_50);
        const bu = Number(c.bollinger_upper);
        const bl = Number(c.bollinger_lower);
        sma20Data.push(Number.isFinite(sma20) ? { x, y: sma20 } : null);
        sma50Data.push(Number.isFinite(sma50) ? { x, y: sma50 } : null);
        bbUpperData.push(Number.isFinite(bu) ? { x, y: bu } : null);
        bbLowerData.push(Number.isFinite(bl) ? { x, y: bl } : null);

        const rv = Number(c.rsi);
        rsiData.push(Number.isFinite(rv) ? { x, y: rv } : null);
        const tbb = Number(c.taker_buy_base_volume);
        const vol = Number(c.volume);
        const pct = (Number.isFinite(tbb) && Number.isFinite(vol) && vol > 0) ? (tbb / vol) * 100 : null;
        takerBuyPctData.push(pct != null && Number.isFinite(pct) ? { x, y: pct } : null);
    }

    const getThemeColor = (name, fallback) => {
        try {
            const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
            return v || fallback;
        } catch {
            return fallback;
        }
    };

    const hexToRgba = (hex, alpha) => {
        const h = (hex || '').trim();
        const m = /^#?([0-9a-fA-F]{6})$/.exec(h);
        if (!m) return `rgba(0,0,0,${alpha})`;
        const intVal = parseInt(m[1], 16);
        const r = (intVal >> 16) & 255;
        const g = (intVal >> 8) & 255;
        const b = intVal & 255;
        return `rgba(${r},${g},${b},${alpha})`;
    };

    const primary = getThemeColor('--primary-color', '#1e40af');
    const success = getThemeColor('--success-color', '#059669');
    const danger = getThemeColor('--danger-color', '#dc2626');
    const dark = getThemeColor('--dark-color', '#1f2937');

    // 指标开关（从 UI 读取；如果 UI 不存在则沿用默认）
    const ind = readAnalysisIndicatorStateFromUI();

    // 三图联动（缩放/拖拽平移）
    let syncLock = false;
    const syncXFrom = (srcChart) => {
        if (syncLock) return;
        try {
            const x = srcChart && srcChart.scales ? srcChart.scales.x : null;
            if (!x) return;
            const min = x.min;
            const max = x.max;
            syncLock = true;
            [analysisCharts.price, analysisCharts.volume, analysisCharts.rsi].forEach(ch => {
                if (!ch || ch === srcChart) return;
                if (!ch.options || !ch.options.scales || !ch.options.scales.x) return;
                ch.options.scales.x.min = min;
                ch.options.scales.x.max = max;
                scheduleChartUpdate(ch, 'none');
            });
        } finally {
            syncLock = false;
        }
    };

    const fitPriceYToVisible = () => {
        const ch = analysisCharts.price;
        if (!ch || !ch.scales || !ch.scales.x || !ch.scales.y) return;
        const x = ch.scales.x;
        const xMin = x.min;
        const xMax = x.max;
        let lo = Infinity;
        let hi = -Infinity;
        // 遍历所有数据集，计算可见范围的高低点（包括 Bollinger Bands 等）
        const datasets = ch.data.datasets || [];
        for (const ds of datasets) {
            // 跳过隐藏的 dataset
            if (ch.isDatasetVisible(ds.index) === false) continue;
            // 跳过辅助线（如选中线）
            if (ds.label === '__SELECTED__') continue;

            const data = ds.data;
            if (!Array.isArray(data)) continue;

            for (const pt of data) {
                // 处理 {x, y} 或 {x, o, h, l, c} 格式
                let valX, valH, valL;

                if (pt && typeof pt === 'object') {
                    valX = pt.x;
                    if (valX < xMin || valX > xMax) continue;

                    // K线数据
                    if ('h' in pt && 'l' in pt) {
                        valH = pt.h;
                        valL = pt.l;
                    }
                    // 线图数据 (SMA, BB 等)
                    else if ('y' in pt) {
                        valH = pt.y;
                        valL = pt.y;
                    }
                } else {
                    continue;
                }

                if (Number.isFinite(valL)) lo = Math.min(lo, valL);
                if (Number.isFinite(valH)) hi = Math.max(hi, valH);
            }
        }

        if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return;
        const pad = (hi - lo) * 0.05; // 增加一点 padding
        ch.options.scales.y.min = lo - pad;
        ch.options.scales.y.max = hi + pad;
        scheduleChartUpdate(ch, 'none');
    };

    const fitVolumeYToVisible = () => {
        const ch = analysisCharts.volume;
        if (!ch || !ch.scales || !ch.scales.x || !ch.scales.y) return;
        const xMin = ch.scales.x.min;
        const xMax = ch.scales.x.max;
        let maxVol = 0;

        // 仅检查第一个 dataset (Volume Bar)
        const ds = ch.data.datasets[0];
        if (ds && Array.isArray(ds.data)) {
            for (const pt of ds.data) {
                if (pt && pt.x >= xMin && pt.x <= xMax) {
                    if (Number.isFinite(pt.y)) maxVol = Math.max(maxVol, pt.y);
                }
            }
        }

        if (maxVol > 0) {
            // Volume 上方留白 20%
            ch.options.scales.y.max = maxVol * 1.2;
            // 保持从 0 开始
            ch.options.scales.y.min = 0;
            scheduleChartUpdate(ch, 'none');
        }
    };

    const onZoomOrPanComplete = (ctx) => {
        const src = ctx && ctx.chart ? ctx.chart : ctx;
        if (!src) return;
        syncXFrom(src);
        // 无论在哪里缩放，都重新适配 Price 和 Volume 的 Y 轴
        fitPriceYToVisible();
        fitVolumeYToVisible();
        updateSelectionMarkers();
    };

    const priceCanvas = document.getElementById('analysis-price-canvas');
    const volumeCanvas = document.getElementById('analysis-volume-canvas');
    const rsiCanvas = document.getElementById('analysis-rsi-canvas');
    if (!priceCanvas || !volumeCanvas || !rsiCanvas) return;

    // 鼠标离开画布时，自动隐藏气泡（未固定时）
    try {
        const onLeave = () => {
            if (!bubblePinned) hideBubble();
        };
        priceCanvas.addEventListener('mouseleave', onLeave);
        volumeCanvas.addEventListener('mouseleave', onLeave);
        rsiCanvas.addEventListener('mouseleave', onLeave);
    } catch {
        // ignore
    }

    try {
        priceCanvas.style.cursor = 'default';
        volumeCanvas.style.cursor = 'default';
        rsiCanvas.style.cursor = 'default';
    } catch {
        // ignore
    }

    const onAnyHover = (chart, evt, activeEls) => {
        if (bubblePinned) return;
        const rec = pickRecordFromClick(chart, evt, activeEls);
        if (rec) {
            showBubble(rec, evt);
        } else {
            hideBubble();
        }
    };

    // chartjs-plugin-zoom（可选）：允许鼠标滚轮缩放
    try {
        if (typeof Chart !== 'undefined' && (window.ChartZoom || window['chartjs-plugin-zoom'])) {
            Chart.register(window.ChartZoom || window['chartjs-plugin-zoom']);
        }
    } catch (e) {
        // ignore
    }

    // 蜡烛图（依赖 chartjs-chart-financial 插件）
    const priceDatasets = [
        {
            label: `${symbol} 蜡烛图`,
            data: candleData,
            color: { up: success, down: danger, unchanged: dark },
            borderColor: { up: success, down: danger, unchanged: dark }
        }
    ];
    if (ind.sma20) {
        priceDatasets.push({
            type: 'line',
            label: 'SMA20',
            data: sma20Data.filter(Boolean),
            parsing: false,
            borderColor: primary,
            borderWidth: 1,
            pointRadius: 0,
            tension: 0
        });
    }
    if (ind.sma50) {
        priceDatasets.push({
            type: 'line',
            label: 'SMA50',
            data: sma50Data.filter(Boolean),
            parsing: false,
            borderColor: dark,
            borderWidth: 1,
            pointRadius: 0,
            tension: 0
        });
    }
    if (ind.boll) {
        priceDatasets.push({
            type: 'line',
            label: 'BOLL_UP',
            data: bbUpperData.filter(Boolean),
            parsing: false,
            borderColor: danger,
            borderWidth: 1,
            pointRadius: 0,
            tension: 0
        });
        priceDatasets.push({
            type: 'line',
            label: 'BOLL_DN',
            data: bbLowerData.filter(Boolean),
            parsing: false,
            borderColor: success,
            borderWidth: 1,
            pointRadius: 0,
            tension: 0
        });
    }
    // 点击反馈：垂直选中线
    priceDatasets.push({
        type: 'line',
        label: '__SELECTED__',
        data: [],
        parsing: false,
        borderColor: dark,
        borderWidth: 1,
        borderDash: [6, 6],
        pointRadius: 0
    });

    analysisCharts.price = new Chart(priceCanvas.getContext('2d'), {
        type: 'candlestick',
        data: { datasets: priceDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: { display: true },
                title: { display: true, text: `${symbol} K线图（${label || appState.selectedTimeframe || 'raw'}）` },
                tooltip: { enabled: false },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'x'
                    },
                    zoom: {
                        wheel: {
                            enabled: true
                        },
                        mode: 'x'
                    },
                    onZoomComplete: onZoomOrPanComplete,
                    onPanComplete: onZoomOrPanComplete
                }
            },
            onHover: (evt, activeEls, chart) => {
                onAnyHover(chart, evt, activeEls);
            },
            parsing: false,
            scales: {
                x: {
                    type: 'linear',
                    ticks: {
                        callback: function (value) {
                            const d = new Date(Number(value));
                            if (Number.isNaN(d.getTime())) return '';
                            return d.toLocaleString('zh-CN');
                        },
                        maxRotation: 45,
                        minRotation: 0
                    }
                },
                y: { title: { display: true, text: '价格' }, beginAtZero: false }
            }
        }
    });

    // 成交量柱状图（与价格图共享 x 轴范围）
    analysisCharts.volume = new Chart(volumeCanvas.getContext('2d'), {
        type: 'bar',
        data: {
            datasets: [
                {
                    label: `${symbol} 成交量`,
                    data: volumeBarData,
                    parsing: false,
                    backgroundColor: hexToRgba(primary, 0.55),
                    borderColor: primary,
                    borderWidth: 1
                },
                {
                    type: 'line',
                    label: `${symbol} 成交额(quote_volume)`,
                    data: quoteVolumeLineData,
                    parsing: false,
                    borderColor: dark,
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0,
                    yAxisID: 'y1'
                },
                {
                    type: 'line',
                    label: '__SELECTED__',
                    data: [],
                    parsing: false,
                    borderColor: dark,
                    borderWidth: 1,
                    borderDash: [6, 6],
                    pointRadius: 0,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: { display: true },
                title: { display: true, text: `成交量/成交额（${symbol}）` },
                tooltip: { enabled: false },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'x'
                    },
                    zoom: {
                        wheel: {
                            enabled: true
                        },
                        mode: 'x'
                    },
                    onZoomComplete: onZoomOrPanComplete,
                    onPanComplete: onZoomOrPanComplete
                }
            },
            onHover: (evt, activeEls) => {
                onAnyHover(analysisCharts.volume, evt, activeEls);
            },
            scales: {
                x: {
                    type: 'linear',
                    display: false
                },
                y: { beginAtZero: true, title: { display: true, text: 'volume' } },
                y1: {
                    beginAtZero: true,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'quote_volume' }
                }
            }
        }
    });

    // RSI / taker-buy% (0-100)
    const rsiDatasets = [];
    if (ind.rsi) {
        rsiDatasets.push({
            label: 'RSI(14)',
            data: rsiData.filter(Boolean),
            parsing: false,
            borderColor: primary,
            borderWidth: 1,
            pointRadius: 0,
            tension: 0
        });
    }
    if (ind.takerBuyPct) {
        rsiDatasets.push({
            label: 'taker_buy% (base)',
            data: takerBuyPctData.filter(Boolean),
            parsing: false,
            borderColor: success,
            borderWidth: 1,
            pointRadius: 0,
            tension: 0
        });
    }
    rsiDatasets.push({
        type: 'line',
        label: '__SELECTED__',
        data: [],
        parsing: false,
        borderColor: dark,
        borderWidth: 1,
        borderDash: [6, 6],
        pointRadius: 0
    });

    analysisCharts.rsi = new Chart(rsiCanvas.getContext('2d'), {
        type: 'line',
        data: {
            datasets: rsiDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: { display: true },
                title: { display: true, text: `RSI / 主动买入占比（${symbol}）` },
                tooltip: { enabled: false },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'x'
                    },
                    zoom: {
                        wheel: {
                            enabled: true
                        },
                        mode: 'x'
                    },
                    onZoomComplete: onZoomOrPanComplete,
                    onPanComplete: onZoomOrPanComplete
                }
            },
            onHover: (evt, activeEls, chart) => {
                onAnyHover(chart, evt, activeEls);
            },
            parsing: false,
            scales: {
                x: {
                    type: 'linear',
                    ticks: {
                        callback: function (value) {
                            const d = new Date(Number(value));
                            if (Number.isNaN(d.getTime())) return '';
                            return d.toLocaleString('zh-CN');
                        },
                        maxRotation: 45,
                        minRotation: 0
                    }
                },
                y: {
                    beginAtZero: true,
                    suggestedMax: 100,
                    suggestedMin: 0
                }
            }
        }
    });

    // 初次渲染：让价格 y 轴贴合可见区间，并同步一次 x 范围
    try {
        fitPriceYToVisible();
        syncXFrom(analysisCharts.price);
        updateSelectionMarkers();
    } catch {
        // ignore
    }
}

// ==================== 交易对搜索 ====================

let allSymbols = [];  // 缓存所有交易对
const symbolListState = {
    loaded: false,
    items: [],
    container: null,
    spacer: null,
    list: null,
    itemHeight: 38,
    overscan: 8
};

function initSymbolVirtualList() {
    const container = document.getElementById('symbol-list-container');
    if (!container) return false;
    if (symbolListState.container) return true;

    container.innerHTML = '';
    const spacer = document.createElement('div');
    spacer.style.position = 'relative';
    spacer.style.height = '0px';
    const list = document.createElement('div');
    list.style.position = 'absolute';
    list.style.left = '0';
    list.style.right = '0';
    list.style.top = '0';
    spacer.appendChild(list);
    container.appendChild(spacer);

    container.addEventListener('scroll', () => {
        renderSymbolListVirtual();
    });

    symbolListState.container = container;
    symbolListState.spacer = spacer;
    symbolListState.list = list;
    return true;
}

function renderSymbolListVirtual() {
    if (!symbolListState.container || !symbolListState.spacer || !symbolListState.list) return;
    const items = symbolListState.items || [];
    const container = symbolListState.container;
    const list = symbolListState.list;
    const spacer = symbolListState.spacer;

    if (items.length === 0) {
        spacer.style.height = '0px';
        list.style.transform = 'translateY(0px)';
        list.innerHTML = '<div class="text-muted p-3">暂无匹配的交易对</div>';
        return;
    }

    const itemHeight = symbolListState.itemHeight;
    const scrollTop = container.scrollTop;
    const viewportHeight = container.clientHeight || 400;
    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - symbolListState.overscan);
    const endIndex = Math.min(items.length, Math.ceil((scrollTop + viewportHeight) / itemHeight) + symbolListState.overscan);

    spacer.style.height = `${items.length * itemHeight}px`;
    list.style.transform = `translateY(${startIndex * itemHeight}px)`;
    list.innerHTML = '';

    const fragment = document.createDocumentFragment();
    const searchTerm = (document.getElementById('symbol-search-input')?.value || '').toUpperCase();

    for (let i = startIndex; i < endIndex; i++) {
        const symbol = items[i];
        const div = document.createElement('div');
        div.className = 'search-result';

        if (appState.selectedSymbols.includes(symbol)) {
            div.classList.add('active');
        }

        if (searchTerm && symbol.toUpperCase().includes(searchTerm)) {
            const regex = new RegExp(`(${searchTerm})`, 'gi');
            div.innerHTML = symbol.replace(regex, '<strong>$1</strong>');
        } else {
            div.textContent = symbol;
        }

        div.addEventListener('click', function () {
            toggleSymbolSelection(symbol);
        });

        fragment.appendChild(div);
    }

    list.appendChild(fragment);

    if (!symbolListState._heightMeasured && list.firstElementChild) {
        const rect = list.firstElementChild.getBoundingClientRect();
        if (rect && rect.height) {
            symbolListState.itemHeight = Math.max(28, Math.round(rect.height));
        }
        symbolListState._heightMeasured = true;
    }
}

async function loadSymbols() {
    try {
        showLoading();
        const data = await fetchJson('/api/search/symbols', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                exchange: appState.selectedExchange,
                market: appState.selectedMarket,
                query: '',
                limit: 1000  // 获取尽可能多的交易对
            })
        }, { timeoutMs: 12000, retries: 1 });
        hideLoading();

        if (data.success) {
            allSymbols = data.symbols || [];
            symbolListState.loaded = true;
            appState.selectedSymbols = [];  // 清空之前的选择
            updateSelectedSymbolsDisplay();
            displaySymbolList(allSymbols);
            showAlert(`已加载 ${allSymbols.length} 个交易对`, 'success');
            addLog(`已从 ${appState.selectedExchange} ${appState.selectedMarket} 加载 ${allSymbols.length} 个交易对`, 'success');
        } else {
            showAlert('加载交易对失败: ' + data.error, 'danger');
            addLog('加载交易对失败: ' + data.error, 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('加载失败:', error);
        const msg = (error && (error.message || error.error)) ? (error.message || error.error) : String(error || 'unknown');
        showAlert('加载交易对失败: ' + msg, 'danger');
        addLog('加载交易对失败: ' + msg, 'error');
    }
}

function filterSymbols(query) {
    if (!query) {
        return allSymbols;
    }

    query = query.toUpperCase();

    // 近似匹配算法：
    // 1. 精确匹配
    // 2. 前缀匹配
    // 3. 包含匹配

    const exact = [];
    const prefix = [];
    const contains = [];

    allSymbols.forEach(symbol => {
        const upper = symbol.toUpperCase();

        if (upper === query) {
            exact.push(symbol);
        } else if (upper.startsWith(query)) {
            prefix.push(symbol);
        } else if (upper.includes(query)) {
            contains.push(symbol);
        }
    });

    return [...exact, ...prefix, ...contains];
}

function displaySymbolList(symbols) {
    const container = document.getElementById('symbol-list-container');
    if (!container) return;

    const shouldShow = symbolListState.loaded || symbols.length > 0;
    container.style.display = shouldShow ? 'block' : 'none';
    if (!shouldShow) {
        return;
    }

    if (!initSymbolVirtualList()) return;

    symbolListState.items = symbols;
    container.scrollTop = 0;
    renderSymbolListVirtual();

    const count = symbols.length > 0 ? `共 ${symbols.length}` : '未找到匹配结果';
    document.getElementById('symbol-count').textContent = count;
}

function toggleSymbolSelection(symbol) {
    if (appState.selectedSymbols.includes(symbol)) {
        appState.selectedSymbols = appState.selectedSymbols.filter(s => s !== symbol);
    } else {
        appState.selectedSymbols.push(symbol);
    }

    updateSelectedSymbolsDisplay();

    renderSymbolListVirtual();
}

function updateSelectedSymbolsDisplay() {
    const container = document.getElementById('selected-symbols');
    container.innerHTML = '';

    if (appState.selectedSymbols.length === 0) {
        container.innerHTML = '<div class="text-muted">尚未选择任何交易对</div>';
    } else {
        appState.selectedSymbols.forEach(symbol => {
            const tag = document.createElement('span');
            tag.className = 'item-tag';
            tag.innerHTML = `
                ${symbol}
                <span class="close-btn" onclick="removeSymbol('${symbol}')">×</span>
            `;
            container.appendChild(tag);
        });
    }

    document.getElementById('selected-count').textContent = appState.selectedSymbols.length;
    renderSymbolListVirtual();
}

function removeSymbol(symbol) {
    appState.selectedSymbols = appState.selectedSymbols.filter(s => s !== symbol);
    updateSelectedSymbolsDisplay();
    renderSymbolListVirtual();
}

function clearSelection() {
    appState.selectedSymbols = [];
    allSymbols = [];
    updateSelectedSymbolsDisplay();
    document.getElementById('symbol-search-input').value = '';
    document.getElementById('symbol-list-container').style.display = 'none';
    document.getElementById('symbol-count').textContent = '点击"加载交易对"按钮获取列表';
    symbolListState.items = [];
    symbolListState.loaded = false;
    addLog('已清除所有选择', 'info');
}

// ==================== 数据获取 ====================

async function fetchData() {
    if (appState.selectedSymbols.length === 0) {
        showAlert('请先选择至少一个交易对', 'warning');
        return;
    }

    if (!appState.dateRange || !appState.dateRange.start || !appState.dateRange.end) {
        showAlert('请先选择开始和结束时间', 'warning');
        return;
    }

    let loadingActive = false;
    try {
        showLoading();
        loadingActive = true;

        // 获取数据限制；为空/无效则传 null，表示使用后端默认值
        const limitInput = document.getElementById('data-limit').value.trim();
        let limit = null;
        if (limitInput !== '') {
            const parsed = parseInt(limitInput, 10);
            if (!Number.isNaN(parsed) && parsed > 0) {
                limit = parsed;
            }
        }

        // 解析时间范围（毫秒时间戳）
        const { startMs, endMs } = getDateRangeMs();
        if (Number.isNaN(startMs) || Number.isNaN(endMs)) {
            hideLoading();
            showAlert('时间格式无效，请重新选择开始和结束时间', 'warning');
            return;
        }

        // 第一步：提交获取任务
        showAlert('正在提交获取任务...', 'info');

        // 提示用户当前解析的时间范围（本地时间）
        const startStr = new Date(startMs).toLocaleString('zh-CN', { hour12: false });
        const endStr = new Date(endMs).toLocaleString('zh-CN', { hour12: false });
        addLog(`准备请求数据，解析为本地时间: ${startStr} -> ${endStr}`, 'info');

        const marketLower = (appState.selectedMarket || 'spot').toLowerCase();
        const body = {
            exchange: appState.selectedExchange,
            market: appState.selectedMarket,
            symbols: appState.selectedSymbols,
            timeframe: appState.selectedTimeframe,
            // limit 传 null 表示使用默认值
            limit: limit,
            start_time: startMs,
            end_time: endMs,
            auto_save: true,
            storage_format: document.getElementById('manage-storage-format')?.value || 'parquet'
        };

        // spot / swap / future / option / margin 支持按 data_type 批量拉取
        if (marketLower === 'spot') {
            body.data_type = appState.selectedSpotDataType || 'ohlcv';
        } else if (marketLower === 'option') {
            body.data_type = appState.selectedOptionDataType || 'ohlcv';
        } else if (marketLower === 'margin') {
            body.data_type = appState.selectedMarginDataType || 'ohlcv';
        } else if (marketLower === 'swap' || marketLower === 'future') {
            body.data_type = appState.selectedDerivDataType || 'ohlcv';
        }

        const response = await fetch('/api/data/batch-fetch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const data = await response.json();

        // 提交任务结束后不再占用全屏遮罩，后续用进度弹窗展示
        hideLoading();
        loadingActive = false;

        if (!data.success) {
            showAlert('提交获取任务失败: ' + data.error, 'danger');
            addLog('提交获取任务失败: ' + data.error, 'error');
            return;
        }

        const taskId = data.task_id;
        addLog(`已提交获取任务，Task ID: ${taskId}`, 'info');

        // 第二步：显示进度模态框
        showFetchProgressModal(taskId, appState.selectedSymbols.length);

        // 第三步：轮询获取进度
        try {
            const taskInfo = await monitorFetchProgress(taskId, appState.selectedSymbols.length);

            console.log('Fetch completed');

            // 显示结果（支持分批下载：与历史结果合并）
            appState.fetchedData = mergeFetchedResults(appState.fetchedData, taskInfo.results);
            refreshAnalysisSymbolOptions();
            displayFetchResults(appState.fetchedData);
            showAlert('✅ 数据获取成功！', 'success');
            addLog(`成功获取 ${Object.keys(taskInfo.results).length} 个交易对的数据（已与历史结果合并）`, 'success');

            // 持久化：刷新页面后仍可恢复
            saveLastMergedResultsToServer(appState.fetchedData);
        } catch (error) {
            console.error('Fetch error:', error);

            // 更新进度条为错误状态
            const progressText = document.getElementById('fetch-progress-text');
            const progressBar = document.getElementById('fetch-progress-bar');

            if (progressText) {
                progressText.innerHTML = `<span class="text-danger"><i class="fas fa-exclamation-circle"></i> 任务失败: ${error.message}</span>`;
            }
            if (progressBar) {
                progressBar.classList.remove('bg-info', 'bg-success', 'progress-bar-animated', 'progress-bar-striped');
                progressBar.classList.add('bg-danger');
                progressBar.style.width = '100%';
            }

            showAlert('获取失败: ' + error.message, 'danger');
            addLog('获取失败: ' + error.message, 'error');
        }
    } catch (error) {
        console.error('获取失败:', error);
        showAlert('获取失败: ' + error.message, 'danger');
        addLog('数据获取失败: ' + error.message, 'error');
    } finally {
        if (loadingActive) {
            hideLoading();
        }
    }
}

function displayFetchResults(results) {
    const tbody = document.getElementById('fetch-results-tbody');
    tbody.innerHTML = '';

    const actions = document.getElementById('fetch-actions');
    if (actions) actions.innerHTML = '';

    const market = (appState.selectedMarket || 'spot').toLowerCase();
    const isSpotNonOhlcv = (market === 'spot') && ((appState.selectedSpotDataType || 'ohlcv') !== 'ohlcv');
    const isOptionNonOhlcv = (market === 'option') && ((appState.selectedOptionDataType || 'ohlcv') !== 'ohlcv');
    const isMarginNonOhlcv = (market === 'margin') && ((appState.selectedMarginDataType || 'ohlcv') !== 'ohlcv');
    const isDerivNonOhlcv = ((market === 'swap' || market === 'future') && ((appState.selectedDerivDataType || 'ohlcv') !== 'ohlcv'));
    const isOhlcvMode = !(market === 'onchain' || market === 'social' || isSpotNonOhlcv || isOptionNonOhlcv || isMarginNonOhlcv || isDerivNonOhlcv);

    const ths = document.querySelectorAll('#fetch-results-table thead th');
    const setHeaders = (names) => {
        if (!ths || ths.length < 7) return;
        for (let i = 0; i < 7; i++) {
            if (names[i] != null) ths[i].textContent = names[i];
        }
    };

    let totalRecords = 0;
    let successCount = 0;

    // 找一个可用于可视化的交易对（OHLCV 或 快照）
    let firstVisualizableSymbol = '';

    if (isOhlcvMode) {
        setHeaders(['交易对', '时间', '开盘价', '最高价', '最低价', '收盘价', '成交量']);
    } else {
        // 在非 OHLCV 模式下，根据第一条成功数据动态确定要展示的 5 个字段
        let sample = null;
        for (const [, result] of Object.entries(results)) {
            if (result && result.status === 'success' && Array.isArray(result.data) && result.data.length > 0) {
                sample = result.data[result.data.length - 1];
                break;
            }
        }
        const metricKeys = sample
            ? Object.keys(sample).filter(k => k !== 'timestamp' && k !== 'datetime').slice(0, 5)
            : [];
        setHeaders([
            '交易对',
            '时间',
            metricKeys[0] || '-',
            metricKeys[1] || '-',
            metricKeys[2] || '-',
            metricKeys[3] || '-',
            metricKeys[4] || '-'
        ]);
        // 把 metricKeys 挂到函数局部，供下面渲染使用
        displayFetchResults._metricKeys = metricKeys;
    }

    for (const [symbol, result] of Object.entries(results)) {
        if (result.status === 'success' && result.data) {
            successCount++;
            totalRecords += result.data.length;

            if (!firstVisualizableSymbol && Array.isArray(result.data) && result.data.length > 0) {
                firstVisualizableSymbol = symbol;
            }

            const fmt = new Intl.DateTimeFormat('zh-CN', {
                timeZone: 'Asia/Shanghai',
                hour12: false,
                year: 'numeric', month: 'numeric', day: 'numeric',
                hour: '2-digit', minute: '2-digit', second: '2-digit'
            });

            if (isOhlcvMode) {
                // 显示最后10条 OHLCV
                result.data.slice(-10).forEach((item) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><strong>${symbol}</strong></td>
                        <td>${item.datetime ? item.datetime : fmt.format(new Date(item.timestamp))}</td>
                        <td>${Number(item.open).toFixed(4)}</td>
                        <td>${Number(item.high).toFixed(4)}</td>
                        <td>${Number(item.low).toFixed(4)}</td>
                        <td>${Number(item.close).toFixed(4)}</td>
                        <td>${Number(item.volume).toFixed(2)}</td>
                    `;
                    tbody.appendChild(row);
                });
            } else {
                const metricKeys = displayFetchResults._metricKeys || [];
                const formatValue = (v) => {
                    if (v == null) return '';
                    if (typeof v === 'number') {
                        if (Number.isInteger(v) || Math.abs(v) >= 1e6) return String(v);
                        return v.toFixed(4);
                    }
                    if (typeof v === 'object') {
                        try {
                            const s = JSON.stringify(v);
                            if (!s) return '';
                            return s.length > 120 ? (s.slice(0, 120) + '…') : s;
                        } catch (e) {
                            return String(v);
                        }
                    }
                    return String(v);
                };

                // 非 OHLCV：每个 symbol 仅展示最后一条快照（或最后10条）
                const rows = result.data.slice(-10);
                rows.forEach((item) => {
                    const row = document.createElement('tr');
                    const t = item.datetime ? item.datetime : fmt.format(new Date(item.timestamp));
                    const cols = metricKeys.map(k => formatValue(item[k]));
                    while (cols.length < 5) cols.push('');
                    row.innerHTML = `
                        <td><strong>${symbol}</strong></td>
                        <td>${t}</td>
                        <td>${cols[0]}</td>
                        <td>${cols[1]}</td>
                        <td>${cols[2]}</td>
                        <td>${cols[3]}</td>
                        <td>${cols[4]}</td>
                    `;
                    tbody.appendChild(row);
                });
            }
        }
    }

    if (successCount === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">无数据</td></tr>';
    }

    // 更新统计信息
    updateFetchStats(results, totalRecords);

    // 非 OHLCV 模式：提供一键复制完整 JSON（表格仍保持截断预览）
    if (actions && !isOhlcvMode && successCount > 0) {
        const btn = document.createElement('button');
        btn.className = 'btn btn-outline-secondary';
        btn.type = 'button';
        btn.innerHTML = '<i class="fas fa-copy"></i> 复制全部JSON';
        btn.addEventListener('click', async () => {
            const text = JSON.stringify(results, null, 2);
            const ok = await copyToClipboard(text);
            if (ok) showAlert('✅ 已复制完整JSON到剪贴板', 'success');
            else showAlert('复制失败：浏览器未允许剪贴板访问', 'warning');
        });
        actions.appendChild(btn);
    }

    // 在“数据获取”页提供一个直达可视化按钮，避免用户找不到入口
    if (actions && firstVisualizableSymbol) {
        const btn = document.createElement('button');
        btn.className = 'btn btn-primary';
        btn.type = 'button';
        btn.innerHTML = isOhlcvMode
            ? '<i class="fas fa-chart-line"></i> 可视化K线（跳转到数据分析）'
            : '<i class="fas fa-table"></i> 可视化快照（跳转到数据分析）';
        btn.addEventListener('click', () => {
            jumpToKlineVisualization(firstVisualizableSymbol);
        });
        actions.appendChild(btn);
    }
}

function mergeFetchedResults(existingResults, incomingResults) {
    const base = (existingResults && typeof existingResults === 'object') ? existingResults : {};
    const inc = (incomingResults && typeof incomingResults === 'object') ? incomingResults : {};
    const merged = { ...base };

    const toMs = (rec) => {
        if (!rec || typeof rec !== 'object') return null;
        if (rec.timestamp != null) {
            const n = Number(rec.timestamp);
            return Number.isFinite(n) ? n : null;
        }
        if (rec.datetime) {
            const ms = Date.parse(rec.datetime);
            return Number.isNaN(ms) ? null : ms;
        }
        return null;
    };

    const mergeByTime = (a, b) => {
        const map = new Map();
        const put = (arr) => {
            if (!Array.isArray(arr)) return;
            for (const rec of arr) {
                const ms = toMs(rec);
                if (ms == null) continue;
                // 后来的覆盖早的（对同一时间点）
                map.set(ms, rec);
            }
        };
        put(a);
        put(b);

        return Array.from(map.entries())
            .sort((x, y) => x[0] - y[0])
            .map(([ms, rec]) => {
                const fixed = { ...rec };
                fixed.timestamp = ms;
                if (!fixed.datetime) fixed.datetime = new Date(ms).toISOString();
                return fixed;
            });
    };

    for (const [symbol, incItem] of Object.entries(inc)) {
        const prevItem = merged[symbol];
        if (!prevItem) {
            merged[symbol] = incItem;
            continue;
        }

        if (
            prevItem && prevItem.status === 'success' && Array.isArray(prevItem.data) &&
            incItem && incItem.status === 'success' && Array.isArray(incItem.data)
        ) {
            merged[symbol] = {
                ...incItem,
                data: mergeByTime(prevItem.data, incItem.data)
            };
            continue;
        }

        // 非可合并结构：用最新结果覆盖
        merged[symbol] = incItem;
    }

    return merged;
}

function jumpToKlineVisualization(symbol) {
    try {
        const select = document.getElementById('analysis-symbol');
        if (select) {
            // 若 options 尚未刷新，尝试补一遍
            if (!Array.from(select.options || []).some(o => o.value === symbol)) {
                refreshAnalysisSymbolOptions();
            }
            select.value = symbol;
        }

        const tabBtn = document.getElementById('data-analysis-tab');
        if (tabBtn) tabBtn.click();

        // 让 tab 切换的 DOM 更新先落地
        setTimeout(() => {
            visualizeKlineFromFetched();
        }, 0);
    } catch (e) {
        showAlert('跳转可视化失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    }
}

function updateFetchStats(results, totalRecords) {
    const statsDiv = document.getElementById('fetch-stats');

    let successCount = 0;
    let errorCount = 0;

    for (const [symbol, result] of Object.entries(results)) {
        if (result.status === 'success') successCount++;
        else if (result.status === 'error') errorCount++;
    }

    statsDiv.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${Object.keys(results).length}</div>
            <div class="stat-label">交易对总数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${successCount}</div>
            <div class="stat-label">成功获取</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${errorCount}</div>
            <div class="stat-label">获取失败</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${totalRecords}</div>
            <div class="stat-label">总记录数</div>
        </div>
    `;
}

// ==================== 数据保存 ====================

async function monitorSaveProgress(taskId, symbolsCount) {
    /**轮询查询保存进度*/
    return new Promise((resolve, reject) => {
        let pollCount = 0;
        const maxPollTime = 3600000;  // 1小时最多轮询
        const startTime = Date.now();

        const poll = async () => {
            try {
                const response = await fetch(`/api/data/save-progress/${taskId}`);
                const data = await response.json();

                if (!data.success) {
                    reject(new Error(data.error || 'Failed to get progress'));
                    return;
                }

                const taskInfo = data.task_info;
                const progress = taskInfo.progress || 0;

                // 更新进度条UI
                updateProgressBar(progress, taskInfo.symbols_processed, symbolsCount);
                addLog(`保存进度: ${progress}% (${taskInfo.symbols_processed}/${symbolsCount})`, 'info');

                // 任务完成或出错
                if (taskInfo.status === 'completed') {
                    updateProgressBar(100, symbolsCount, symbolsCount);
                    addLog('数据保存完成！', 'success');
                    resolve(taskInfo);
                } else if (taskInfo.status === 'error') {
                    reject(new Error(taskInfo.error || 'Save task failed'));
                } else if (Date.now() - startTime > maxPollTime) {
                    reject(new Error('Save task timeout (> 1 hour)'));
                } else {
                    // 继续轮询，间隔1秒
                    pollCount++;
                    setTimeout(poll, 1000);
                }
            } catch (error) {
                reject(error);
            }
        };

        poll();
    });
}

function updateProgressBar(progress, processed, total) {
    /**更新进度条显示*/
    const progressBar = document.getElementById('save-progress-bar');
    const progressText = document.getElementById('save-progress-text');

    if (progressBar) {
        progressBar.style.width = progress + '%';
        progressBar.textContent = progress + '%';
    }

    if (progressText) {
        progressText.textContent = `保存进度: ${progress}% (${processed}/${total})`;
    }
}

function showSaveProgressModal(taskId, symbolsCount) {
    /**显示保存进度模态框*/
    const modal = document.createElement('div');
    modal.id = 'save-progress-modal';
    modal.className = 'modal fade show';
    modal.style.display = 'block';
    modal.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">保存数据进度</h5>
                </div>
                <div class="modal-body">
                    <div class="progress" style="height: 25px;">
                        <div id="save-progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" 
                             role="progressbar" style="width: 0%">0%</div>
                    </div>
                    <p id="save-progress-text" class="mt-2 text-center">保存进度: 0% (0/${symbolsCount})</p>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
    return modal;
}

async function monitorFetchProgress(taskId, symbolsCount) {
    /**轮询查询获取数据进度*/
    return new Promise((resolve, reject) => {
        let pollCount = 0;
        const maxPollTime = 3600000;  // 1小时最多轮询
        const maxPollCount = 3600;  // 最多轮询3600次（1小时，1秒1次）
        const startTime = Date.now();

        const poll = async () => {
            try {
                const response = await fetch(`/api/data/fetch-progress/${taskId}`);
                const data = await response.json();

                if (!data.success) {
                    reject(new Error(data.error || 'Failed to get progress'));
                    return;
                }

                const taskInfo = data.task_info;
                const progress = taskInfo.progress || 0;
                const status = taskInfo.status;
                const processed = Number(taskInfo.symbols_processed || 0);
                const resultsObj = taskInfo.results && typeof taskInfo.results === 'object' ? taskInfo.results : {};
                const resultsCount = Object.keys(resultsObj).length;

                console.log(`Task ${taskId} status: ${status}, progress: ${progress}%`);

                // 更新进度条UI
                updateFetchProgressBar(progress, taskInfo.symbols_processed, symbolsCount);
                addLog(`获取进度: ${progress}% (${taskInfo.symbols_processed}/${symbolsCount})`, 'info');

                // 任务完成或出错
                if (status === 'completed') {
                    updateFetchProgressBar(100, symbolsCount, symbolsCount);
                    addLog('数据获取完成！', 'success');
                    console.log('Task completed, resolving...');
                    // 兜底：避免 overlay 意外残留
                    hideLoading();
                    resolve(taskInfo);
                    return;
                } else if ((progress >= 100 || processed >= symbolsCount) && resultsCount >= symbolsCount) {
                    // 兜底：极少数情况下后台线程已跑完但 status 未及时切换
                    updateFetchProgressBar(100, symbolsCount, symbolsCount);
                    addLog('数据获取完成（状态兜底判断）', 'success');
                    hideLoading();
                    resolve(taskInfo);
                    return;
                } else if (status === 'error') {
                    // 兜底：避免 overlay 意外残留
                    hideLoading();
                    reject(new Error(taskInfo.error || 'Fetch task failed'));
                    return;
                } else if (Date.now() - startTime > maxPollTime) {
                    hideLoading();
                    reject(new Error('Fetch task timeout (> 1 hour)'));
                    return;
                } else if (pollCount >= maxPollCount) {
                    hideLoading();
                    reject(new Error(`Fetch task timeout (polled ${pollCount} times)`));
                    return;
                } else {
                    // 继续轮询，间隔1秒
                    pollCount++;
                    setTimeout(poll, 1000);
                }
            } catch (error) {
                console.error('Poll error:', error);
                hideLoading();
                reject(error);
            }
        };

        poll();
    });
}

function updateFetchProgressBar(progress, processed, total) {
    /**更新获取数据进度条显示*/
    const progressBar = document.getElementById('fetch-progress-bar');
    const progressText = document.getElementById('fetch-progress-text');

    if (progressBar) {
        progressBar.style.width = progress + '%';
        progressBar.textContent = progress + '%';

        // 动态颜色
        if (progress >= 100) {
            progressBar.classList.remove('bg-info', 'progress-bar-striped', 'progress-bar-animated');
            progressBar.classList.add('bg-success');
        } else {
            progressBar.classList.remove('bg-success');
            progressBar.classList.add('bg-info', 'progress-bar-striped', 'progress-bar-animated');
        }
    }

    if (progressText) {
        // 根据进度显示不同阶段
        let phaseText = '';
        if (progress >= 100) {
            phaseText = '<i class="fas fa-check-circle text-success me-1"></i>获取完成';
        } else if (progress <= 50) {
            phaseText = '正在获取数据...';
        } else {
            phaseText = '正在处理数据...';
        }
        progressText.innerHTML = `${phaseText} <span class="badge bg-secondary">${progress}%</span> (${processed}/${total})`;
    }
}

function showFetchProgressModal(taskId, symbolsCount) {
    /** 显示内嵌进度条（替代原模态框） */
    const container = document.getElementById('fetch-stats');
    if (!container) return;

    // 清空旧内容
    container.innerHTML = '';

    const progressDiv = document.createElement('div');
    progressDiv.id = 'fetch-progress-container';
    progressDiv.className = 'p-3 mb-3 border rounded bg-light';
    progressDiv.style.borderLeft = '4px solid var(--info-color)';
    progressDiv.innerHTML = `
        <h6 class="mb-2"><i class="fas fa-spinner fa-spin me-2"></i>正在执行任务...</h6>
        <div class="progress" style="height: 25px;">
            <div id="fetch-progress-bar" class="progress-bar progress-bar-striped progress-bar-animated bg-info" 
                 role="progressbar" style="width: 0%">0%</div>
        </div>
        <p id="fetch-progress-text" class="mt-2 mb-0 text-muted small">准备开始获取 ${symbolsCount} 个交易对的数据...</p>
    `;

    // 插入到开头
    container.prepend(progressDiv);
    return progressDiv;
}

async function saveData() {
    if (appState.selectedSymbols.length === 0) {
        showAlert('请先选择至少一个交易对', 'warning');
        return;
    }

    try {
        const { startMs, endMs } = getDateRangeMs();

        // 第一步：提交保存任务，获得 task_id
        showAlert('正在提交保存任务...', 'info');
        const storageFormat = document.getElementById('manage-storage-format')?.value || 'parquet';
        const response = await fetch('/api/data/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                market: appState.selectedMarket,
                exchange: appState.selectedExchange,
                symbols: appState.selectedSymbols,
                timeframe: appState.selectedTimeframe,
                start_time: Number.isNaN(startMs) ? null : startMs,
                end_time: Number.isNaN(endMs) ? null : endMs,
                storage_format: storageFormat
            })
        });

        const data = await response.json();

        if (!data.success) {
            showAlert('提交保存任务失败: ' + data.error, 'danger');
            addLog('提交保存任务失败: ' + data.error, 'error');
            return;
        }

        const taskId = data.task_id;
        addLog(`已提交保存任务，Task ID: ${taskId}`, 'info');

        // 第二步：显示进度模态框
        showSaveProgressModal(taskId, appState.selectedSymbols.length);

        // 第三步：轮询获取进度
        try {
            const taskInfo = await monitorSaveProgress(taskId, appState.selectedSymbols.length);

            // 关闭进度模态框
            const modal = document.getElementById('save-progress-modal');
            if (modal) modal.remove();

            showAlert('✅ 数据保存成功！', 'success');
            addLog(`已保存 ${appState.selectedSymbols.length} 个交易对的数据`, 'success');
        } catch (error) {
            const modal = document.getElementById('save-progress-modal');
            if (modal) modal.remove();

            showAlert('保存失败: ' + error.message, 'danger');
            addLog('保存失败: ' + error.message, 'error');
        }
    } catch (error) {
        console.error('保存失败:', error);
        showAlert('保存失败: ' + error.message, 'danger');
        addLog('保存失败: ' + error.message, 'error');
    }
}

// ==================== 数据管理 ====================

let manageAllSymbols = [];  // 缓存管理模块的交易对
let manageSelectedSymbols = [];

async function fetchLocalIndex(params = {}) {
    const q = new URLSearchParams();
    Object.entries(params || {}).forEach(([k, v]) => {
        if (v === undefined || v === null || v === '') return;
        q.set(k, String(v));
    });
    const res = await fetch(`/api/local/index?${q.toString()}`);
    const data = await res.json();
    if (!data || !data.success) {
        throw new Error((data && data.error) ? data.error : '本地索引加载失败');
    }
    return data;
}

function renderLocalIndexTable(data) {
    const summaryEl = document.getElementById('local-index-summary');
    const tbody = document.getElementById('local-index-tbody');

    const entries = Array.isArray(data && data.entries) ? data.entries : [];
    if (summaryEl) {
        const n = entries.length;
        summaryEl.textContent = `已加载 ${n} 条（generated_at=${data.generated_at || '-'}）`;
    }

    if (!tbody) return;
    if (entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">无匹配本地数据</td></tr>';
        return;
    }

    const rowsHtml = entries.slice(0, 300).map(e => {
        const mkt = e.market || '-';
        const ex = e.exchange || '-';
        const sym = e.symbol || e.symbol_key || '-';
        const tf = e.timeframe || '-';
        const startDate = e.start_date ? e.start_date.substring(0, 16) : '-';
        const endDate = e.end_date ? e.end_date.substring(0, 16) : '-';
        const count = e.count ? e.count.toLocaleString() : '-';
        const format = e.format ? `<span class="badge bg-${e.format === 'parquet' ? 'info' : 'secondary'}">${e.format}</span>` : '-';

        // Escape file path for JS string
        const safePath = (e.file_path || '').replace(/\\/g, '\\\\').replace(/'/g, "\\'");

        return `
            <tr>
                <td>${format} ${mkt}</td>
                <td>${ex}</td>
                <td><strong>${sym}</strong></td>
                <td>${tf}</td>
                <td>${startDate} -> ${endDate}</td>
                <td>${count} 条</td>
                <td>
                   <button class="btn btn-sm btn-outline-primary" onclick="checkDataIntegrity('${safePath}', '${tf}')" title="检查数据完整度">
                        <i class="fas fa-heartbeat"></i> 体检
                   </button>
                </td>
                <td>
                    <div class="d-flex flex-wrap gap-1">
                        <button class="btn btn-sm btn-outline-warning" onclick="convertLocalDataFile('${safePath}', '${e.format || ''}')" title="转换文件格式">
                            <i class="fas fa-exchange-alt"></i> 转换
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteLocalDataFile('${safePath}', '${sym}', '${tf}')" title="删除本地数据文件">
                            <i class="fas fa-trash"></i> 删除
                        </button>
                    </div>
                </td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = rowsHtml;
}

// 供HTML onclick调用
window.checkDataIntegrity = async function (filePath, timeframe) {
    if (!filePath) return;

    const modalEl = document.getElementById('integrityModal');
    const bodyEl = document.getElementById('integrityModalBody');
    const modal = new bootstrap.Modal(modalEl);

    // Reset loading state
    bodyEl.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2 text-muted">正在进行深度扫描，请耐心等待...</p>
        </div>
    `;

    modal.show();

    try {
        const response = await fetch('/api/local/data-integrity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: filePath, timeframe: timeframe })
        });

        const data = await response.json();

        if (!data.success) {
            bodyEl.innerHTML = `<div class="alert alert-danger">检查失败: ${data.error}</div>`;
            return;
        }

        const r = data.report;
        let gapsHtml = '';

        if (r.gaps_count > 0 && r.head_gaps) {
            const safePathForRepair = filePath.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
            gapsHtml = `
                <h6 class="mt-3 text-warning">检测到 ${r.gaps_count} 个断档 (显示前 ${r.head_gaps.length} 个):</h6>
                <div class="table-responsive" style="max-height: 200px; overflow-y: auto;">
                    <table class="table table-xs table-sm table-bordered">
                        <thead><tr><th>开始 (UTC)</th><th>结束 (UTC)</th><th>缺失时长</th><th>操作</th></tr></thead>
                        <tbody>
                            ${r.head_gaps.map(g => `
                                <tr>
                                    <td>${g.start.replace('T', ' ')}</td>
                                    <td>${g.end.replace('T', ' ')}</td>
                                    <td class="text-danger">${g.duration_str}</td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-primary py-0" 
                                            onclick="repairDataGap('${safePathForRepair}', '${timeframe}', '${g.start}', '${g.end}')"
                                            title="尝试从交易所补录此段数据">
                                            <i class="fas fa-tools"></i> 补录
                                        </button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        } else if (r.status === 'ok') {
            gapsHtml = `<div class="alert alert-success mt-3"><i class="fas fa-check-circle"></i> 数据连续性完美，无缺失。</div>`;
        }

        const statusBadge = r.status === 'ok' ? '<span class="badge bg-success">健康</span>' : '<span class="badge bg-warning text-dark">异常</span>';
        const coverageClass = r.coverage_pct > 95 ? 'text-success' : (r.coverage_pct > 50 ? 'text-warning' : 'text-danger');

        bodyEl.innerHTML = `
            <div class="container-fluid px-0">
                <div class="row mb-3">
                    <div class="col-6"><strong>文件名:</strong><br><small class="text-muted">${r.file_name}</small></div>
                    <div class="col-3"><strong>状态:</strong><br>${statusBadge}</div>
                    <div class="col-3"><strong>覆盖率:</strong><br><span class="${coverageClass} fw-bold">${r.coverage_pct}%</span></div>
                </div>
                <div class="row mb-3 bg-light p-2 rounded mx-0">
                     <div class="col-6">
                        <small class="text-muted">实际行数 / 理论行数</small><br>
                        <strong>${r.total_rows.toLocaleString()}</strong> / ${r.expected_rows.toLocaleString()}
                     </div>
                     <div class="col-6">
                        <small class="text-muted">时间范围 (文件内)</small><br>
                        ${r.start_time.replace('T', ' ')} <br>⬇<br> ${r.end_time.replace('T', ' ')}
                     </div>
                </div>
                ${gapsHtml}
            </div>
        `;

    } catch (e) {
        console.error(e);
        bodyEl.innerHTML = `<div class="alert alert-danger">请求错误: ${e.message}</div>`;
    }
};

window.deleteLocalDataFile = async function (filePath, symbol, timeframe) {
    if (!filePath) return;

    const label = [symbol, timeframe].filter(Boolean).join(' / ');
    const tip = label ? `将删除：${label}` : '将删除所选文件';
    const input = prompt(`${tip}\n\n为防止误操作，请输入 DELETE 确认：`, '');
    if (input !== 'DELETE') {
        showAlert('已取消删除', 'info');
        return;
    }

    try {
        showLoading();
        const resp = await fetch('/api/local/delete-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: filePath })
        });

        const data = await resp.json();
        if (!data || !data.success) {
            throw new Error((data && data.error) ? data.error : '删除失败');
        }

        showAlert('✅ 本地数据已删除', 'success');
        await refreshLocalIndexTable();
    } catch (e) {
        console.error(e);
        showAlert('删除失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    } finally {
        hideLoading();
    }
};

window.convertLocalDataFile = async function (filePath, currentFormat) {
    if (!filePath) return;
    const targetFormat = document.getElementById('local-convert-format')?.value || 'parquet';
    if (currentFormat && String(currentFormat).toLowerCase() === String(targetFormat).toLowerCase()) {
        showAlert('当前文件已是目标格式', 'info');
        return;
    }

    const input = prompt('将转换为 ' + targetFormat.toUpperCase() + ' 格式。\n\n为防止误操作，请输入 CONVERT 确认：', '');
    if (input !== 'CONVERT') {
        showAlert('已取消转换', 'info');
        return;
    }

    try {
        showLoading();
        const resp = await fetch('/api/local/convert-format', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: filePath, target_format: targetFormat })
        });
        const data = await resp.json();
        if (!data || !data.success) {
            throw new Error((data && data.error) ? data.error : '转换失败');
        }
        showAlert('✅ 转换完成', 'success');
        await refreshLocalIndexTable();
    } catch (e) {
        console.error(e);
        showAlert('转换失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    } finally {
        hideLoading();
    }
};

window.repairDataGap = async function (filePath, timeframe, startTime, endTime) {
    const input = prompt(
        `确定要尝试补录 ${startTime} 到 ${endTime} 的数据吗？\n\n` +
        `✅ 点击确定开始补录（默认：优先 binance，不足自动回退其它交易所并验证覆盖率）。\n` +
        `🔄 若需指定交易所，请输入交易所代码（如 okx, bybit）。\n` +
        `🌐 若需使用代理，格式为：exchange|proxy_url\n例如: binance|http://127.0.0.1:7890`,
        ""
    );

    if (input === null) return; // Cancelled

    let sourceExchange = input.trim();
    let proxyUrl = "";

    if (sourceExchange.includes('|')) {
        const parts = sourceExchange.split('|');
        sourceExchange = parts[0].trim();
        proxyUrl = parts[1].trim();
    }

    try {
        showLoading();

        const payload = {
            file_path: filePath,
            timeframe: timeframe,
            start_time: startTime,
            end_time: endTime,
            source_exchange: sourceExchange
        };

        if (proxyUrl) {
            payload.proxy = proxyUrl;
        }

        const resp = await fetch('/api/local/repair-gap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await resp.json();

        if (data.success) {
            showAlert(`补录成功: 抓取并合并了 ${data.fetched_rows} 条数据`, 'success');
            await checkDataIntegrity(filePath, timeframe);
        } else {
            showAlert(`补录失败: ${data.error}`, 'danger');
        }
    } catch (e) {
        console.error(e);
        showAlert(`请求错误: ${e.message}`, 'danger');
    } finally {
        hideLoading();
    }
};

async function refreshLocalIndexTable() {
    try {
        showLoading();
        const market = document.getElementById('manage-market')?.value || 'spot';
        const exchange = document.getElementById('manage-exchange')?.value || 'binance';
        const timeframe = document.getElementById('manage-timeframe')?.value || '';
        const includeRange = document.getElementById('local-index-include-range')?.checked ? '1' : '';

        const data = await fetchLocalIndex({
            market,
            exchange,
            timeframe,
            include_range: includeRange,
            max_files: 20000,
            force: '1'
        });

        renderLocalIndexTable(data);
        showAlert('✅ 本地索引已刷新', 'success');
    } catch (e) {
        console.error('刷新本地索引失败:', e);
        showAlert('刷新本地索引失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    } finally {
        hideLoading();
    }
}

async function loadAnalysisLocalData() {
    try {
        showLoading();
        const market = document.getElementById('analysis-market-select')?.value || 'spot';
        const exchange = document.getElementById('analysis-exchange-select')?.value || 'binance';

        const data = await fetchLocalIndex({
            market,
            exchange,
            timeframe: '',
            include_range: '0',
            max_files: 20000,
            force: '1'
        });

        const symbols = Array.isArray(data.symbols) ? data.symbols : [];
        const symbolSelect = document.getElementById('analysis-symbol');
        if (symbolSelect && symbols.length > 0) {
            symbolSelect.innerHTML = '';
            symbols.forEach(sym => {
                const opt = document.createElement('option');
                opt.value = sym;
                opt.textContent = sym;
                symbolSelect.appendChild(opt);
            });

            const statusEl = document.getElementById('analysis-local-status');
            if (statusEl) {
                statusEl.textContent = `已加载 ${symbols.length} 个交易对`;
            }

            showAlert(`已从本地加载 ${symbols.length} 个交易对`, 'success');
        } else {
            showAlert('本地无可用数据，请先保存数据到存储', 'warning');
            const statusEl = document.getElementById('analysis-local-status');
            if (statusEl) {
                statusEl.textContent = '无数据';
            }
        }
    } catch (e) {
        console.error('加载本地分析数据失败:', e);
        showAlert('加载本地分析数据失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    } finally {
        hideLoading();
    }
}

async function loadAnalysisDataFromLocal(symbol) {
    try {
        showLoading();
        const market = document.getElementById('analysis-market-select')?.value || 'spot';
        const exchange = document.getElementById('analysis-exchange-select')?.value || 'binance';

        const resp = await fetch('/api/local/data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                market,
                exchange,
                symbol,
                timeframe: ''
            })
        });

        const respData = await resp.json();
        if (!respData || !respData.success) {
            throw new Error(respData && respData.error ? respData.error : '加载失败');
        }

        const rows = Array.isArray(respData.data) ? respData.data : [];
        if (!appState.fetchedData || typeof appState.fetchedData !== 'object') {
            appState.fetchedData = {};
        }
        appState.fetchedData[symbol] = {
            status: 'success',
            data: rows,
            count: rows.length
        };

        displayFetchResults(appState.fetchedData);

        const hint = document.getElementById('analysis-mode-hint');
        if (hint) {
            hint.textContent = `本地加载: ${respData.meta ? respData.meta.source : 'local'} | ${rows.length} 条数据 | ${respData.meta ? respData.meta.timeframe : '?'} | 文件: ${respData.meta && respData.meta.file_path ? respData.meta.file_path : '?'}`;
        }

        showAlert(`已加载本地数据: ${symbol} (${rows.length} 条)`, 'success');
        addLog(`本地数据已加载: ${market}/${exchange}/${symbol}`, 'info');

        // 自动切换到可视化标签页
        const analysisTab = document.getElementById('data-analysis-tab');
        if (analysisTab) {
            analysisTab.click();
        }
    } catch (e) {
        console.error('加载本地数据失败:', e);
        showAlert('加载本地数据失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    } finally {
        hideLoading();
    }
}

async function loadAnalysisSelectedFromLocal() {
    const select = document.getElementById('analysis-symbol');
    const symbols = getSelectedValues(select);
    if (!symbols.length) {
        showAlert('请先在“本地交易对”中选择至少一个', 'warning');
        return;
    }

    try {
        showLoading();
        let ok = 0;
        for (const sym of symbols) {
            try {
                await loadAnalysisDataFromLocal(sym);
                ok += 1;
            } catch (e) {
                // loadAnalysisDataFromLocal 内部已提示
                console.warn('批量加载单个失败:', sym, e);
            }
        }
        showAlert(`批量加载完成：成功 ${ok}/${symbols.length}`, 'success');
    } finally {
        hideLoading();
    }
}

async function incrementalFetchSelectedLocal() {
    const select = document.getElementById('analysis-symbol');
    const symbols = getSelectedValues(select);
    if (!symbols.length) {
        showAlert('请先选择要增量补齐的交易对（可多选）', 'warning');
        return;
    }

    try {
        showLoading();
        const market = document.getElementById('analysis-market-select')?.value || 'spot';
        const exchange = document.getElementById('analysis-exchange-select')?.value || 'binance';

        const resp = await fetch('/api/local/incremental-fetch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                market,
                exchange,
                symbols,
                include_internal: true,
                include_tail: true,
                dry_run: false,
                max_gaps: 200
            })
        });
        const data = await resp.json();
        if (!data || !data.success) {
            throw new Error(data && data.error ? data.error : '增量补齐失败');
        }

        const results = data.results || {};
        let ok = 0;
        let totalFetched = 0;
        for (const [sym, r] of Object.entries(results)) {
            if (r && r.success) {
                ok += 1;
                totalFetched += Number(r.fetched_rows || 0);
            }
        }
        showAlert(`增量补齐完成：成功 ${ok}/${symbols.length}，新增拉取 ${totalFetched} 条`, 'success');
        addLog(`增量补齐完成：market=${market}, exchange=${exchange}, symbols=${symbols.length}`, 'info');
    } catch (e) {
        console.error('增量补齐失败:', e);
        showAlert('增量补齐失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    } finally {
        hideLoading();
    }
}

async function exportStandardizedSelectedLocal() {
    const select = document.getElementById('analysis-symbol');
    const symbols = getSelectedValues(select);
    if (!symbols.length) {
        showAlert('请先选择要导出的交易对（可多选）', 'warning');
        return;
    }

    try {
        showLoading();
        const market = document.getElementById('analysis-market-select')?.value || 'spot';
        const exchange = document.getElementById('analysis-exchange-select')?.value || 'binance';

        const resp = await fetch('/api/local/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                market,
                exchange,
                symbols,
                format: 'csv'
            })
        });
        const data = await resp.json();
        if (!data || !data.success) {
            throw new Error(data && data.error ? data.error : '导出失败');
        }

        showAlert(`导出成功：${data.rows || 0} 行，已开始下载`, 'success');
        if (data.download_url) {
            window.location.href = data.download_url;
        }
    } catch (e) {
        console.error('标准化导出失败:', e);
        showAlert('标准化导出失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    } finally {
        hideLoading();
    }
}

async function loadManageSymbolsFromLocal() {
    try {
        showLoading();
        const market = document.getElementById('manage-market')?.value || 'spot';
        const exchange = document.getElementById('manage-exchange')?.value || 'binance';
        const timeframe = document.getElementById('manage-timeframe')?.value || '';
        const includeRange = document.getElementById('local-index-include-range')?.checked ? '1' : '';

        const data = await fetchLocalIndex({
            market,
            exchange,
            timeframe,
            include_range: includeRange,
            max_files: 20000,
            force: '1'
        });

        manageAllSymbols = Array.isArray(data.symbols) ? data.symbols : [];
        manageSelectedSymbols = [];
        updateManageSelectedSymbolsDisplay();
        displayManageSymbolList(manageAllSymbols);
        renderLocalIndexTable(data);

        showAlert(`已从本地加载 ${manageAllSymbols.length} 个交易对`, 'success');
    } catch (e) {
        console.error('从本地加载交易对失败:', e);
        showAlert('从本地加载交易对失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    } finally {
        hideLoading();
    }
}

async function loadManageSymbols() {
    try {
        showLoading();
        const response = await fetch('/api/search/symbols', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                exchange: document.getElementById('manage-exchange').value,
                market: document.getElementById('manage-market').value,
                query: '',
                limit: 1000
            })
        });

        const data = await response.json();
        hideLoading();

        if (data.success) {
            manageAllSymbols = data.symbols || [];
            manageSelectedSymbols = [];
            updateManageSelectedSymbolsDisplay();
            displayManageSymbolList(manageAllSymbols);
            showAlert(`已加载 ${manageAllSymbols.length} 个交易对`, 'success');
        } else {
            showAlert('加载交易对失败: ' + data.error, 'danger');
        }
    } catch (error) {
        hideLoading();
        console.error('加载失败:', error);
        const msg = (error && (error.message || error.error)) ? (error.message || error.error) : String(error || 'unknown');
        showAlert('加载交易对失败: ' + msg, 'danger');
    }
}

function filterManageSymbols(query) {
    if (!query) {
        return manageAllSymbols;
    }

    query = query.toUpperCase();
    const exact = [];
    const prefix = [];
    const contains = [];

    manageAllSymbols.forEach(symbol => {
        const upper = symbol.toUpperCase();
        if (upper === query) {
            exact.push(symbol);
        } else if (upper.startsWith(query)) {
            prefix.push(symbol);
        } else if (upper.includes(query)) {
            contains.push(symbol);
        }
    });

    return [...exact, ...prefix, ...contains];
}

function displayManageSymbolList(symbols) {
    const resultsDiv = document.getElementById('manage-symbol-results');
    resultsDiv.style.display = symbols.length > 0 ? 'block' : 'none';
    resultsDiv.innerHTML = '';

    if (symbols.length === 0) {
        resultsDiv.innerHTML = '<div class="text-muted p-3">暂无匹配的交易对</div>';
        return;
    }

    symbols.slice(0, 200).forEach(symbol => {
        const div = document.createElement('div');
        div.className = 'search-result';

        if (manageSelectedSymbols.includes(symbol)) {
            div.classList.add('active');
        }

        div.textContent = symbol;
        div.addEventListener('click', function () {
            toggleManageSymbolSelection(symbol);
        });

        resultsDiv.appendChild(div);
    });
}

async function searchSymbolsForManage(query) {
    if (manageAllSymbols.length === 0) {
        addLog('自动从本地索引加载交易对...', 'info');
        await loadManageSymbolsFromLocal();
        if (manageAllSymbols.length === 0) {
            showAlert('本地索引为空：请先点击“从本地加载”或确认已保存过数据', 'warning');
            return;
        }
    }

    const filtered = filterManageSymbols(query);
    displayManageSymbolList(filtered);
}

function toggleManageSymbolSelection(symbol) {
    if (manageSelectedSymbols.includes(symbol)) {
        manageSelectedSymbols = manageSelectedSymbols.filter(s => s !== symbol);
    } else {
        manageSelectedSymbols.push(symbol);
    }

    updateManageSelectedSymbolsDisplay();
}

function updateManageSelectedSymbolsDisplay() {
    const container = document.getElementById('manage-selected-symbols');
    container.innerHTML = '';

    if (manageSelectedSymbols.length === 0) {
        container.innerHTML = '<div class="text-muted">尚未选择任何交易对</div>';
    } else {
        manageSelectedSymbols.forEach(symbol => {
            const tag = document.createElement('span');
            tag.className = 'item-tag';
            tag.innerHTML = `
                ${symbol}
                <span class="close-btn" onclick="removeManageSymbol('${symbol}')">×</span>
            `;
            container.appendChild(tag);
        });
    }

    const countElement = document.getElementById('manage-selected-count');
    if (countElement) {
        countElement.textContent = manageSelectedSymbols.length;
    }
}

function removeManageSymbol(symbol) {
    manageSelectedSymbols = manageSelectedSymbols.filter(s => s !== symbol);
    updateManageSelectedSymbolsDisplay();
}

function clearManageSelection() {
    manageSelectedSymbols = [];
    manageAllSymbols = [];
    updateManageSelectedSymbolsDisplay();
    document.getElementById('manage-symbol-search').value = '';
    document.getElementById('manage-symbol-results').style.display = 'none';
    addLog('已清除所有管理模块选择', 'info');
}

function getUtc8MidnightMs() {
    const now = new Date();
    const utcMs = now.getTime() + (now.getTimezoneOffset() * 60000);
    const utc8 = new Date(utcMs + 8 * 3600000);
    utc8.setHours(0, 0, 0, 0);
    // convert back to UTC ms
    return utc8.getTime() - 8 * 3600000;
}

async function autoUpdateManageSymbolsToMidnight() {
    if (manageSelectedSymbols.length === 0) {
        showAlert('请先选择至少一个交易对', 'warning');
        return;
    }

    try {
        showLoading();
        const market = document.getElementById('manage-market')?.value || 'spot';
        const exchange = document.getElementById('manage-exchange')?.value || 'binance';
        const timeframe = document.getElementById('manage-timeframe')?.value || '';

        const endMs = getUtc8MidnightMs();

        const resp = await fetch('/api/local/incremental-fetch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                market,
                exchange,
                symbols: manageSelectedSymbols,
                timeframe: timeframe || null,
                include_internal: true,
                include_tail: true,
                dry_run: false,
                max_gaps: 200,
                end_ms: endMs
            })
        });

        const data = await resp.json();
        if (!data || !data.success) {
            throw new Error(data && data.error ? data.error : '自动更新失败');
        }

        const results = data.results || {};
        const total = Object.keys(results).length;
        const ok = Object.values(results).filter(r => r && r.success).length;
        showAlert(`✅ 已更新 ${ok}/${total} 个交易对（截止到今日 0 点）`, 'success');
        addLog(`自动更新完成：${ok}/${total}（截止到今日0点，UTC+8）`, 'success');
    } catch (e) {
        console.error('自动更新失败:', e);
        showAlert('自动更新失败: ' + (e && e.message ? e.message : String(e)), 'danger');
        addLog('自动更新失败: ' + (e && e.message ? e.message : String(e)), 'error');
    } finally {
        hideLoading();
    }
}

function renderManageClassification(items) {
    const tbody = document.getElementById('manage-classify-tbody');
    const summaryEl = document.getElementById('manage-classify-summary');
    if (!tbody || !summaryEl) return;

    if (!Array.isArray(items) || items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">暂无分类结果</td></tr>';
        summaryEl.innerHTML = '<li class="text-muted">暂无</li>';
        return;
    }

    const sorted = items.slice().sort((a, b) => {
        const ga = String(a.group_id || '');
        const gb = String(b.group_id || '');
        if (ga !== gb) return ga.localeCompare(gb);
        return String(a.symbol || '').localeCompare(String(b.symbol || ''));
    });

    const rows = sorted.map(it => {
        const symbol = it.symbol || '';
        const base = it.base || '';
        const tradeType = it.trade_type || '';
        const quoteTypeLabel = it.quote_type_label || '';
        const groupId = it.group_id || '';
        return `<tr>
            <td>${symbol}</td>
            <td>${base}</td>
            <td>${tradeType}</td>
            <td>${quoteTypeLabel}</td>
            <td><span class="text-muted">${groupId}</span></td>
        </tr>`;
    }).join('');

    tbody.innerHTML = rows;

    const groupCount = {};
    sorted.forEach(it => {
        const gid = it.group_id || 'UNKNOWN';
        groupCount[gid] = (groupCount[gid] || 0) + 1;
    });

    const summaryItems = Object.entries(groupCount)
        .sort((a, b) => b[1] - a[1])
        .map(([gid, cnt]) => `<li>${gid} <span class="text-muted">(${cnt})</span></li>`)
        .join('');

    summaryEl.innerHTML = summaryItems || '<li class="text-muted">暂无</li>';
}

async function classifyManageSymbols() {
    if (manageSelectedSymbols.length === 0) {
        showAlert('请先选择至少一个交易对', 'warning');
        return;
    }

    try {
        showLoading();
        const market = document.getElementById('manage-market')?.value || 'spot';
        const exchange = document.getElementById('manage-exchange')?.value || 'binance';

        const resp = await fetch('/api/symbols/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                market,
                exchange,
                symbols: manageSelectedSymbols
            })
        });

        const data = await resp.json();
        if (!data || !data.success) {
            throw new Error(data && data.error ? data.error : '分类失败');
        }

        renderManageClassification(data.items || []);
        showAlert(`✅ 已生成 ${data.count || 0} 个交易对分类`, 'success');
    } catch (e) {
        console.error('交易对分类失败:', e);
        showAlert('交易对分类失败: ' + (e && e.message ? e.message : String(e)), 'danger');
    } finally {
        hideLoading();
    }
}

async function saveToStorage() {
    if (manageSelectedSymbols.length === 0) {
        showAlert('请先选择至少一个交易对', 'warning');
        return;
    }

    try {
        // 获取时间框架，如果未定义则使用 1h
        const timeframe = document.getElementById('manage-timeframe')?.value || '1h';

        // 第一步：提交保存任务
        showAlert('正在提交保存任务...', 'info');
        const storageFormat = document.getElementById('manage-storage-format')?.value || 'parquet';
        const response = await fetch('/api/data/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                market: document.getElementById('manage-market').value,
                exchange: document.getElementById('manage-exchange').value,
                symbols: manageSelectedSymbols,
                timeframe: timeframe,
                storage_format: storageFormat
            })
        });

        const data = await response.json();

        if (!data.success) {
            showAlert('提交保存任务失败: ' + data.error, 'danger');
            addLog('提交保存任务失败: ' + data.error, 'error');
            return;
        }

        const taskId = data.task_id;
        addLog(`已提交保存任务，Task ID: ${taskId}`, 'info');

        // 第二步：显示进度模态框
        showSaveProgressModal(taskId, manageSelectedSymbols.length);

        // 第三步：轮询获取进度
        try {
            const taskInfo = await monitorSaveProgress(taskId, manageSelectedSymbols.length);

            // 关闭进度模态框
            const modal = document.getElementById('save-progress-modal');
            if (modal) modal.remove();

            showAlert('✅ 数据已保存到存储！', 'success');
            addLog(`已将 ${manageSelectedSymbols.length} 个交易对的数据保存到存储`, 'success');
        } catch (error) {
            const modal = document.getElementById('save-progress-modal');
            if (modal) modal.remove();

            showAlert('保存失败: ' + error.message, 'danger');
            addLog('保存失败: ' + error.message, 'error');
        }
    } catch (error) {
        console.error('保存失败:', error);
        showAlert('保存失败: ' + error.message, 'danger');
        addLog('保存失败: ' + error.message, 'error');
    }
}

function exportData() {
    if (Object.keys(appState.fetchedData).length === 0) {
        showAlert('暂无数据可导出', 'warning');
        return;
    }

    const marketLower = (appState.selectedMarket || 'spot').toLowerCase();
    let dataType = 'ohlcv';
    if (marketLower === 'spot') dataType = appState.selectedSpotDataType || 'ohlcv';
    else if (marketLower === 'option') dataType = appState.selectedOptionDataType || 'ohlcv';
    else if (marketLower === 'margin') dataType = appState.selectedMarginDataType || 'ohlcv';
    else if (marketLower === 'swap' || marketLower === 'future') dataType = appState.selectedDerivDataType || 'ohlcv';
    else if (marketLower === 'onchain' || marketLower === 'social') dataType = 'snapshot';

    const rows = [];
    const toMs = (item) => {
        if (!item) return null;
        if (item.timestamp != null) {
            const n = Number(item.timestamp);
            return Number.isFinite(n) ? n : null;
        }
        if (item.datetime) {
            const ms = Date.parse(item.datetime);
            return Number.isNaN(ms) ? null : ms;
        }
        return null;
    };

    for (const [symbol, result] of Object.entries(appState.fetchedData)) {
        if (!result || result.status !== 'success' || !Array.isArray(result.data)) continue;
        for (const item of result.data) {
            const ts = toMs(item);
            const base = {
                exchange: appState.selectedExchange,
                market: appState.selectedMarket,
                timeframe: appState.selectedTimeframe,
                data_type: dataType,
                symbol: symbol,
                timestamp: ts,
                datetime: item && item.datetime ? item.datetime : (ts != null ? new Date(ts).toISOString() : null),
            };

            if (item && typeof item === 'object') {
                for (const [k, v] of Object.entries(item)) {
                    if (k === 'timestamp' || k === 'datetime') continue;
                    if (v == null) {
                        base[k] = null;
                    } else if (typeof v === 'object') {
                        try {
                            base[k] = JSON.stringify(v);
                        } catch {
                            base[k] = String(v);
                        }
                    } else {
                        base[k] = v;
                    }
                }
            }
            rows.push(base);
        }
    }

    if (rows.length === 0) {
        showAlert('暂无可导出的成功数据', 'warning');
        return;
    }

    // 生成列头：固定元信息在前，其余字段按字母序追加
    const fixedCols = ['exchange', 'market', 'timeframe', 'data_type', 'symbol', 'timestamp', 'datetime'];
    const extraColsSet = new Set();
    rows.forEach(r => Object.keys(r).forEach(k => { if (!fixedCols.includes(k)) extraColsSet.add(k); }));
    const extraCols = Array.from(extraColsSet).sort();
    const cols = fixedCols.concat(extraCols);

    const csvEscape = (val) => {
        if (val == null) return '';
        const s = String(val);
        if (s.includes('"') || s.includes(',') || s.includes('\n') || s.includes('\r')) {
            return '"' + s.replaceAll('"', '""') + '"';
        }
        return s;
    };

    let csv = cols.join(',') + '\n';
    for (const r of rows) {
        csv += cols.map(c => csvEscape(r[c])).join(',') + '\n';
    }

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    const safe = (s) => String(s || '').replaceAll(/[^a-zA-Z0-9._-]+/g, '_');
    link.setAttribute('download', `crypto_data_${safe(appState.selectedExchange)}_${safe(appState.selectedMarket)}_${safe(dataType)}_${new Date().getTime()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    showAlert('数据已导出！', 'success');
    addLog('数据已导出为 CSV 文件', 'success');
}

// ==================== 系统监控 ====================

async function updateSystemStatus() {
    try {
        const data = await fetchJson('/api/status', {}, { timeoutMs: 8000, retries: 1 });

        document.getElementById('stat-fetchers').textContent = data.cached_fetchers;
        document.getElementById('stat-managers').textContent = data.cached_managers;
        document.getElementById('stat-version').textContent = data.version;

        addLog('系统状态已更新', 'info');
    } catch (error) {
        console.error('状态更新失败:', error);
        addLog('状态更新失败: ' + error.message, 'error');
    }
}

function updateSystemTime() {
    const now = new Date();
    const timeStr = now.toLocaleString('zh-CN');
    const timeElement = document.getElementById('system-time');
    if (timeElement) {
        timeElement.textContent = timeStr;
    }
}

// ==================== Chart 更新节流 ====================

const chartUpdateQueue = new Map();
let chartUpdateTimer = null;

function scheduleChartUpdate(chart, mode = 'none') {
    if (!chart) return;
    chartUpdateQueue.set(chart, mode);
    if (chartUpdateTimer) return;
    chartUpdateTimer = requestAnimationFrame(() => {
        const batch = Array.from(chartUpdateQueue.entries());
        chartUpdateQueue.clear();
        chartUpdateTimer = null;
        batch.forEach(([ch, md]) => {
            try {
                ch.update(md || 'none');
            } catch {
                // ignore
            }
        });
    });
}

// ==================== 网络请求封装（超时/重试） ====================

async function fetchWithTimeout(url, options = {}, cfg = {}) {
    const {
        timeoutMs = 12000,
        retries = 1,
        retryDelayMs = 600,
        retryOn = [408, 429, 500, 502, 503, 504]
    } = cfg || {};

    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const resp = await fetch(url, { ...options, signal: controller.signal });
            clearTimeout(timer);

            if (retryOn.includes(resp.status) && attempt < retries) {
                await new Promise(r => setTimeout(r, retryDelayMs * (attempt + 1)));
                continue;
            }

            return resp;
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

async function fetchJson(url, options = {}, cfg = {}) {
    const resp = await fetchWithTimeout(url, options, cfg);
    const data = await resp.json();
    if (!resp.ok) {
        const msg = data?.error || `HTTP ${resp.status}`;
        throw new Error(msg);
    }
    return data;
}

// ==================== UI 辅助函数 ====================

function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alert-container');
    const alertHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-circle' : 'info-circle'}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    alertContainer.insertAdjacentHTML('beforeend', alertHTML);

    // 5秒后自动关闭
    setTimeout(() => {
        const alerts = alertContainer.querySelectorAll('.alert');
        if (alerts.length > 0) {
            alerts[0].remove();
        }
    }, 5000);
}

function showLoading() {
    const el = document.getElementById('loading-overlay');
    if (!el) return;
    el.classList.add('active');
}

function hideLoading() {
    const el = document.getElementById('loading-overlay');
    if (!el) return;
    el.classList.remove('active');
}

// ==================== 深度图（Order Book） ====================

let depthChart = null;

async function renderDepthChartForSymbol(symbol) {
    const canvas = document.getElementById('depth-chart');
    const hint = document.getElementById('depth-chart-hint');
    if (!canvas) return;
    const exchange = appState.selectedExchange || 'binance';
    if (hint) hint.textContent = `${exchange} / ${symbol}`;

    try {
        const data = await fetchJson(`/api/spot/orderbook?exchange=${exchange}&symbol=${encodeURIComponent(symbol)}&limit=50`, {}, { timeoutMs: 6000, retries: 1 });
        if (!data.success || !data.data) throw new Error(data.error || 'orderbook failed');

        const bids = data.data.bids || [];
        const asks = data.data.asks || [];

        const buildCumulative = (rows, desc = true) => {
            const sorted = rows.slice().sort((a, b) => desc ? b[0] - a[0] : a[0] - b[0]);
            let cum = 0;
            return sorted.map(([price, amount]) => {
                cum += Number(amount);
                return { x: Number(price), y: cum };
            });
        };

        const bidSeries = buildCumulative(bids, true);
        const askSeries = buildCumulative(asks, false);

        if (!depthChart) {
            depthChart = new Chart(canvas.getContext('2d'), {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Bids',
                            data: bidSeries,
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16,185,129,0.15)',
                            fill: true,
                            stepped: true
                        },
                        {
                            label: 'Asks',
                            data: askSeries,
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239,68,68,0.15)',
                            fill: true,
                            stepped: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: true }
                    },
                    scales: {
                        x: { type: 'linear', title: { display: true, text: 'Price' } },
                        y: { title: { display: true, text: 'Cumulative Volume' } }
                    },
                    animation: { duration: 0 }
                }
            });
        } else {
            depthChart.data.datasets[0].data = bidSeries;
            depthChart.data.datasets[1].data = askSeries;
            depthChart.update('none');
        }
    } catch (e) {
        if (hint) hint.textContent = `深度图加载失败: ${e?.message || String(e)}`;
    }
}

function addLog(message, level = 'info') {
    const timestamp = new Date().toLocaleString('zh-CN');
    const logEntry = `[${timestamp}] [${level.toUpperCase()}] ${message}`;
    appState.systemLogs.push(logEntry);

    // 只保留最近100条日志
    if (appState.systemLogs.length > 100) {
        appState.systemLogs.shift();
    }

    // 更新日志显示（增量渲染，避免每次全量重绘）
    const logsDiv = document.getElementById('system-logs');
    if (logsDiv) {
        let color = '#6b7280';
        if (logEntry.includes('ERROR')) color = '#dc2626';
        else if (logEntry.includes('SUCCESS')) color = '#059669';
        else if (logEntry.includes('WARNING')) color = '#d97706';

        const row = document.createElement('div');
        row.style.color = color;
        row.textContent = logEntry;
        logsDiv.appendChild(row);

        while (logsDiv.children.length > 100) {
            logsDiv.removeChild(logsDiv.firstChild);
        }

        logsDiv.scrollTop = logsDiv.scrollHeight;
    }
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// ==================== ML / Research ====================

function mlSetTaskUI(taskId, status, error, logs) {
    const idEl = document.getElementById('ml-task-id');
    const stEl = document.getElementById('ml-task-status');
    const errEl = document.getElementById('ml-task-error');
    const logsEl = document.getElementById('ml-task-logs');
    const progEl = document.getElementById('ml-progress');
    if (idEl) idEl.textContent = taskId || '-';
    if (stEl) stEl.textContent = status || '-';
    if (errEl) errEl.textContent = error || '-';
    if (logsEl) {
        const lines = Array.isArray(logs) ? logs : [];
        logsEl.textContent = lines.join('\n');
        logsEl.scrollTop = logsEl.scrollHeight;
    }
    if (progEl) {
        const pct = Number(progEl.dataset.pct || 0);
        progEl.style.width = `${pct}%`;
        progEl.textContent = `${pct}%`;
    }
}

function mlStopPolling() {
    if (appState.mlPollTimer) {
        clearInterval(appState.mlPollTimer);
        appState.mlPollTimer = null;
    }
}

function mlStartPolling(taskId) {
    mlStopPolling();
    if (!taskId) return;
    appState.mlTaskId = taskId;
    const progEl = document.getElementById('ml-progress');
    if (progEl) progEl.dataset.pct = '0';
    mlSetTaskUI(taskId, 'polling', '-', []);
    appState.mlPollTimer = setInterval(async () => {
        await refreshMLTask();
    }, 3000);
}

async function loadMLConfig() {
    try {
        const data = await fetchJson('/api/ml/config', {}, { timeoutMs: 8000, retries: 1 });
        if (!data.success) return;
        const cfg = data.config || {};
        const pathEl = document.getElementById('ml-data-path');
        if (pathEl && !pathEl.value) pathEl.value = cfg.data_path || '';
        const horizonEl = document.getElementById('ml-horizon');
        if (horizonEl && cfg.horizon_steps) horizonEl.value = String(cfg.horizon_steps);
        const maxRowsEl = document.getElementById('ml-max-rows');
        if (maxRowsEl && cfg.max_rows) maxRowsEl.value = String(cfg.max_rows);
        const maxRunEl = document.getElementById('ml-max-runtime');
        if (maxRunEl && cfg.max_runtime_seconds) maxRunEl.value = String(cfg.max_runtime_seconds);
    } catch (e) {
        console.error('ML config load failed', e);
    }
}

function collectMLOverrides() {
    const overrides = {};
    const pathEl = document.getElementById('ml-data-path');
    if (pathEl && pathEl.value) overrides.data_path = pathEl.value.trim();
    const horizonEl = document.getElementById('ml-horizon');
    if (horizonEl && horizonEl.value) overrides.horizon_steps = Number(horizonEl.value);
    const maxRowsEl = document.getElementById('ml-max-rows');
    if (maxRowsEl && maxRowsEl.value) overrides.max_rows = Number(maxRowsEl.value);
    const maxRunEl = document.getElementById('ml-max-runtime');
    if (maxRunEl && maxRunEl.value) overrides.max_runtime_seconds = Number(maxRunEl.value);
    if (document.getElementById('ml-quick-mode')?.checked) overrides.quick_mode = true;

    const exps = [];
    if (document.getElementById('ml-exp-baseline')?.checked) {
        exps.push({ name: 'ml_baseline', model_type: 'ml' });
    }
    if (document.getElementById('ml-exp-pca')?.checked) {
        exps.push({ name: 'ml_pca', model_type: 'ml', use_pca: true });
    }
    if (document.getElementById('ml-exp-ic')?.checked) {
        exps.push({ name: 'ml_ic_weight', model_type: 'ml', use_ic_weight: true });
    }
    if (document.getElementById('ml-exp-lstm')?.checked) {
        exps.push({ name: 'lstm', model_type: 'deep', deep_model: 'lstm' });
    }
    if (document.getElementById('ml-exp-transformer')?.checked) {
        exps.push({ name: 'transformer', model_type: 'deep', deep_model: 'transformer' });
    }
    if (exps.length > 0) overrides.experiments = exps;

    return overrides;
}

async function runMLPipeline() {
    try {
        const payload = collectMLOverrides();
        const data = await fetchJson('/api/ml/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }, { timeoutMs: 15000, retries: 1 });
        if (!data.success) throw new Error(data.error || 'ML run failed');
        mlStartPolling(data.task_id);
        showAlert('ML 任务已提交', 'success');
    } catch (e) {
        showAlert('ML 任务提交失败: ' + (e?.message || String(e)), 'danger');
    }
}

async function refreshMLTask() {
    if (!appState.mlTaskId) return;
    try {
        const data = await fetchJson(`/api/ml/task/${encodeURIComponent(appState.mlTaskId)}`, {}, { timeoutMs: 8000, retries: 1 });
        if (!data.success) throw new Error(data.error || 'Task not found');
        const task = data.task || {};
        const progEl = document.getElementById('ml-progress');
        if (progEl && typeof task.progress === 'number') {
            progEl.dataset.pct = String(Math.max(0, Math.min(100, task.progress)));
        }
        mlSetTaskUI(task.task_id, task.status, task.error, task.logs);
        if (task.status === 'completed' || task.status === 'error') {
            mlStopPolling();
            loadMLSummary();
        }
    } catch (e) {
        mlStopPolling();
        mlSetTaskUI(appState.mlTaskId, 'error', e?.message || String(e), []);
    }
}

async function cancelMLTask() {
    if (!appState.mlTaskId) {
        showAlert('没有运行中的任务', 'warning');
        return;
    }
    try {
        const data = await fetchJson(`/api/ml/cancel/${encodeURIComponent(appState.mlTaskId)}`, { method: 'POST' }, { timeoutMs: 8000, retries: 1 });
        if (!data.success) throw new Error(data.error || 'Cancel failed');
        mlStopPolling();
        mlSetTaskUI(appState.mlTaskId, 'cancelled', 'Cancelled by user', []);
        showAlert('任务已取消', 'success');
    } catch (e) {
        showAlert('取消失败: ' + (e?.message || String(e)), 'danger');
    }
}

function renderMLSummary(summary) {
    const container = document.getElementById('ml-summary-container');
    if (!container) return;
    const entries = summary ? Object.entries(summary) : [];
    if (entries.length === 0) {
        container.innerHTML = '<div class="text-muted">暂无结果</div>';
        return;
    }

    const html = entries.map(([expName, exp]) => {
        const metrics = exp.metrics || {};
        const backtest = exp.backtest || {};
        const equityPaths = exp.equity_paths || {};
        const modelRows = Object.keys(metrics).map(m => {
            const mt = metrics[m] || {};
            const bt = backtest[m] || {};
            const eq = equityPaths[m] || '';
            return `
                <tr>
                    <td>${expName}</td>
                    <td>${m}</td>
                    <td>${(mt.mae ?? '-').toFixed ? mt.mae.toFixed(6) : (mt.mae ?? '-')}</td>
                    <td>${(mt.rmse ?? '-').toFixed ? mt.rmse.toFixed(6) : (mt.rmse ?? '-')}</td>
                    <td>${(mt.r2 ?? '-').toFixed ? mt.r2.toFixed(4) : (mt.r2 ?? '-')}</td>
                    <td>${(mt.dir_acc ?? '-').toFixed ? mt.dir_acc.toFixed(4) : (mt.dir_acc ?? '-')}</td>
                    <td>${(mt.dir_f1 ?? '-').toFixed ? mt.dir_f1.toFixed(4) : (mt.dir_f1 ?? '-')}</td>
                    <td>${(mt.ic ?? '-').toFixed ? mt.ic.toFixed(4) : (mt.ic ?? '-')}</td>
                    <td>${(mt.icir ?? '-').toFixed ? mt.icir.toFixed(4) : (mt.icir ?? '-')}</td>
                    <td>${(bt.ls_ret ?? '-').toFixed ? bt.ls_ret.toFixed(6) : (bt.ls_ret ?? '-')}</td>
                    <td>${(bt.sharpe ?? '-').toFixed ? bt.sharpe.toFixed(4) : (bt.sharpe ?? '-')}</td>
                    <td>${eq ? `<span class="text-muted small">${eq}</span>` : '-'}</td>
                </tr>
            `;
        }).join('');

        return modelRows;
    }).join('');

    container.innerHTML = `
        <table class="table table-sm table-striped align-middle">
            <thead>
                <tr>
                    <th>Experiment</th>
                    <th>Model</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R2</th>
                    <th>Dir Acc</th>
                    <th>Dir F1</th>
                    <th>IC</th>
                    <th>ICIR</th>
                    <th>LS Ret</th>
                    <th>Sharpe</th>
                    <th>Equity Path</th>
                </tr>
            </thead>
            <tbody>
                ${html}
            </tbody>
        </table>
    `;
}

async function loadMLSummary() {
    try {
        const data = await fetchJson('/api/ml/results', {}, { timeoutMs: 8000, retries: 1 });
        if (!data.success) throw new Error(data.error || 'Failed to load ML results');
        if (!data.available) {
            renderMLSummary({});
            return;
        }
        renderMLSummary(data.summary || {});
    } catch (e) {
        console.error('Failed to load ML summary', e);
    }
}

// ==================== 链上看板 ====================

function getOnchainCommonPayload() {
    return {
        network: document.getElementById('onchain-network')?.value || 'ethereum',
        chain: document.getElementById('onchain-chain')?.value || 'mainnet',
        simulation: !!document.getElementById('onchain-simulation')?.checked
    };
}

function parseJsonInput(text, label) {
    const raw = (text || '').trim();
    if (!raw) return null;
    try {
        return JSON.parse(raw);
    } catch (e) {
        showAlert(`${label} 不是有效 JSON`, 'danger');
        return null;
    }
}

function renderOnchainResult(resultElId, data) {
    const el = document.getElementById(resultElId);
    if (!el) return;
    el.textContent = JSON.stringify(data, null, 2);
}

const onchainCharts = {};

function destroyOnchainChart(key) {
    if (onchainCharts[key]) {
        onchainCharts[key].destroy();
        delete onchainCharts[key];
    }
}

function extractSeries(rows) {
    if (!Array.isArray(rows) || rows.length === 0) return null;
    const timeKeys = ['timestamp', 'date', 'block_time', 'time', 'day'];
    const sample = rows[0];
    const valueKey = Object.keys(sample || {}).find(k => typeof sample[k] === 'number');
    if (!valueKey) return null;
    const series = [];
    rows.forEach(r => {
        let t = null;
        for (const k of timeKeys) {
            if (r[k]) { t = r[k]; break; }
        }
        if (t && typeof r[valueKey] === 'number') {
            series.push({ t, v: r[valueKey] });
        }
    });
    if (!series.length) return null;
    return { key: valueKey, points: series };
}

function renderOnchainCharts(kind, data) {
    const chartMap = {
        exchange_flow: 'onchain-flow-chart',
        address_behavior: 'onchain-behavior-chart',
        large_moves: 'onchain-large-chart',
        mev: 'onchain-mev-chart',
        gas: 'onchain-gas-chart',
        protocol: 'onchain-protocol-chart',
        capital_cycle: 'onchain-cycle-chart',
        token_distribution: 'onchain-token-chart',
        nft: 'onchain-nft-chart',
        price_relation: 'onchain-price-chart'
    };
    const canvasId = chartMap[kind];
    if (!canvasId) return;
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;

    destroyOnchainChart(canvasId);

    const ctx = canvas.getContext('2d');
    let chartCfg = null;

    if (data && Array.isArray(data.rows)) {
        const series = extractSeries(data.rows);
        if (series) {
            chartCfg = {
                type: 'line',
                data: {
                    labels: series.points.map(p => String(p.t)),
                    datasets: [{
                        label: series.key,
                        data: series.points.map(p => p.v),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.15)',
                        tension: 0.2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: true } },
                    scales: { x: { display: false } }
                }
            };
        }
    }

    if (!chartCfg) {
        if (kind === 'exchange_flow') {
            const summary = data.summary || (data.results && data.results[0]) || {};
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['inflow', 'outflow', 'net'],
                    datasets: [{
                        label: 'flow',
                        data: [summary.inflow || 0, summary.outflow || 0, summary.net_flow || 0],
                        backgroundColor: ['#10b981', '#ef4444', '#3b82f6']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        } else if (kind === 'address_behavior') {
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['active', 'new', 'retained'],
                    datasets: [{
                        label: 'count',
                        data: [data.active_count || 0, data.new_count || 0, data.retained_count || 0],
                        backgroundColor: ['#3b82f6', '#10b981', '#f59e0b']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        } else if (kind === 'large_moves') {
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['large_tx', 'top_from_share', 'top_to_share'],
                    datasets: [{
                        label: 'value',
                        data: [
                            data.large_tx_count || 0,
                            (data.from_concentration?.top_share || 0) * 100,
                            (data.to_concentration?.top_share || 0) * 100
                        ],
                        backgroundColor: ['#6366f1', '#f97316', '#22c55e']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        } else if (kind === 'mev') {
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['rows'],
                    datasets: [{
                        label: 'rows',
                        data: [Array.isArray(data.rows) ? data.rows.length : 0],
                        backgroundColor: ['#0ea5e9']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        } else if (kind === 'gas') {
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['gas_price_gwei', 'gas_utilization'],
                    datasets: [{
                        label: 'value',
                        data: [data.gas_price?.gas_price_gwei || 0, (data.gas_utilization || 0) * 100],
                        backgroundColor: ['#14b8a6', '#eab308']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        } else if (kind === 'protocol') {
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['rows'],
                    datasets: [{
                        label: 'rows',
                        data: [Array.isArray(data.rows) ? data.rows.length : 0],
                        backgroundColor: ['#a855f7']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        } else if (kind === 'capital_cycle') {
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['samples'],
                    datasets: [{
                        label: 'samples',
                        data: [Array.isArray(data.flow_samples) ? data.flow_samples.length : 0],
                        backgroundColor: ['#38bdf8']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        } else if (kind === 'token_distribution') {
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['whale_ratio', 'top_share'],
                    datasets: [{
                        label: 'ratio(%)',
                        data: [
                            (data.whale_ratio_sample || 0) * 100,
                            (data.holder_concentration?.top_share || 0) * 100
                        ],
                        backgroundColor: ['#ef4444', '#3b82f6']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        } else if (kind === 'nft') {
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['rows'],
                    datasets: [{
                        label: 'rows',
                        data: [Array.isArray(data.rows) ? data.rows.length : 0],
                        backgroundColor: ['#f472b6']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        } else if (kind === 'price_relation') {
            chartCfg = {
                type: 'bar',
                data: {
                    labels: ['corr_ret', 'corr_vol'],
                    datasets: [{
                        label: 'corr',
                        data: [data.corr_metric_ret || 0, data.corr_metric_vol || 0],
                        backgroundColor: ['#22c55e', '#f59e0b']
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            };
        }
    }

    if (chartCfg) {
        onchainCharts[canvasId] = new Chart(ctx, chartCfg);
    }
}

function extractRowsFromOnchainData(data) {
    if (!data || typeof data !== 'object') return [];
    if (Array.isArray(data.rows)) return data.rows;
    if (Array.isArray(data.results)) return data.results;
    if (Array.isArray(data.flow_samples)) return data.flow_samples;
    if (data.summary && typeof data.summary === 'object') return [data.summary];
    if (data.gas_price && typeof data.gas_price === 'object') return [data.gas_price];
    if (data.holder_concentration && typeof data.holder_concentration === 'object') return [data.holder_concentration];
    return [];
}

function renderOnchainTable(kind, data) {
    const tableIdMap = {
        exchange_flow: 'onchain-flow-table',
        address_behavior: 'onchain-behavior-table',
        large_moves: 'onchain-large-table',
        mev: 'onchain-mev-table',
        gas: 'onchain-gas-table',
        protocol: 'onchain-protocol-table',
        capital_cycle: 'onchain-cycle-table',
        token_distribution: 'onchain-token-table',
        nft: 'onchain-nft-table',
        price_relation: 'onchain-price-table'
    };
    const tableId = tableIdMap[kind];
    if (!tableId) return;
    const container = document.getElementById(tableId);
    if (!container) return;

    const rows = extractRowsFromOnchainData(data);
    if (!rows.length) {
        container.innerHTML = '<div class="text-muted small">暂无可表格化数据</div>';
        return;
    }

    const keys = Array.from(rows.reduce((set, row) => {
        Object.keys(row || {}).forEach(k => set.add(k));
        return set;
    }, new Set()));

    const head = keys.map(k => `<th>${k}</th>`).join('');
    const body = rows.slice(0, 50).map(r => {
        const tds = keys.map(k => `<td>${r[k] === undefined ? '' : String(r[k])}</td>`).join('');
        return `<tr>${tds}</tr>`;
    }).join('');

    container.innerHTML = `
        <div class="table-responsive">
            <table class="table table-sm table-striped align-middle">
                <thead><tr>${head}</tr></thead>
                <tbody>${body}</tbody>
            </table>
        </div>
    `;
}

function convertRowsToCsv(rows) {
    if (!rows || !rows.length) return '';
    const keys = Array.from(rows.reduce((set, row) => {
        Object.keys(row || {}).forEach(k => set.add(k));
        return set;
    }, new Set()));
    const escape = (val) => {
        const str = val === undefined || val === null ? '' : String(val);
        if (str.includes('"') || str.includes(',') || str.includes('\n')) {
            return '"' + str.replace(/"/g, '""') + '"';
        }
        return str;
    };
    const header = keys.map(escape).join(',');
    const lines = rows.map(r => keys.map(k => escape(r[k])).join(','));
    return [header, ...lines].join('\n');
}

function downloadOnchainCsv(kind) {
    const payload = appState.onchainResults[kind];
    if (!payload) {
        showAlert('没有可下载的结果', 'warning');
        return;
    }
    const rows = extractRowsFromOnchainData(payload);
    if (!rows.length) {
        showAlert('结果为空，无法导出 CSV', 'warning');
        return;
    }
    const csv = convertRowsToCsv(rows);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.href = url;
    a.download = `onchain_${kind}_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

function copyOnchainResult(kind) {
    const payload = appState.onchainResults[kind];
    if (!payload) {
        showAlert('没有可复制的结果', 'warning');
        return;
    }
    const text = JSON.stringify(payload, null, 2);
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text)
            .then(() => showAlert('已复制 JSON', 'success'))
            .catch(() => showAlert('复制失败', 'danger'));
    } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        ta.remove();
        showAlert('已复制 JSON', 'success');
    }
}

function downloadOnchainResult(kind) {
    const payload = appState.onchainResults[kind];
    if (!payload) {
        showAlert('没有可下载的结果', 'warning');
        return;
    }
    const text = JSON.stringify(payload, null, 2);
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.href = url;
    a.download = `onchain_${kind}_${ts}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

async function runOnchainAnalyze(kind, payload, resultElId) {
    const el = document.getElementById(resultElId);
    if (el) el.textContent = '运行中...';
    try {
        showLoading();
        const data = await fetchJson('/api/onchain/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ kind, ...payload })
        }, { timeoutMs: 30000, retries: 0 });
        if (!data.success) throw new Error(data.error || '请求失败');
        appState.onchainResults[kind] = data.data || {};
        renderOnchainResult(resultElId, data.data || {});
        renderOnchainCharts(kind, data.data || {});
        renderOnchainTable(kind, data.data || {});
    } catch (e) {
        if (el) el.textContent = '';
        showAlert(`链上分析失败: ${e?.message || e}`, 'danger');
    } finally {
        hideLoading();
    }
}

console.log('应用脚本已加载');


// ==================== ���׶Լ��� ====================

document.addEventListener('DOMContentLoaded', function () {
    loadAvailablePairs();

    // --- AlphaGen Event Bindings ---
    // Data Prep
    const btnLoadMerged = document.getElementById('btn-alphagen-load-files');
    if (btnLoadMerged) btnLoadMerged.addEventListener('click', alphagenLoadMergedFiles);

    const btnInferSwap = document.getElementById('btn-alphagen-infer-swap');
    if (btnInferSwap) btnInferSwap.addEventListener('click', alphagenInferSwapAux);

    const btnBuildPanel = document.getElementById('btn-alphagen-build-panel');
    if (btnBuildPanel) btnBuildPanel.addEventListener('click', alphagenBuildPanel);

    const btnRecommendPanel = document.getElementById('btn-alphagen-recommend-panel');
    if (btnRecommendPanel) btnRecommendPanel.addEventListener('click', alphagenApplyRecommendedPanel);

    const mergedSelect = document.getElementById('alphagen-merged-select');
    if (mergedSelect) {
        mergedSelect.addEventListener('change', () => {
            const opt = mergedSelect.selectedOptions?.[0];
            alphagenRenderMergedStatsFromOption(opt);
        });
    }

    const btnCheckPanel = document.getElementById('btn-alphagen-check-panel');
    if (btnCheckPanel) btnCheckPanel.addEventListener('click', alphagenCheckPanel);

    // Train & Eval
    const btnTrain = document.getElementById('btn-alphagen-train');
    if (btnTrain) btnTrain.addEventListener('click', alphagenStartTrain);

    const btnRefresh = document.getElementById('btn-alphagen-refresh');
    if (btnRefresh) btnRefresh.addEventListener('click', alphagenRefreshJob);

    const btnCancel = document.getElementById('btn-alphagen-cancel');
    if (btnCancel) btnCancel.addEventListener('click', alphagenCancelJob);

    const btnEval = document.getElementById('btn-alphagen-eval');
    if (btnEval) btnEval.addEventListener('click', alphagenEvaluateExpr);

    const evalModeEl = document.getElementById('alphagen-eval-mode');
    if (evalModeEl) {
        evalModeEl.addEventListener('change', async () => {
            const mode = evalModeEl.value;
            const panel = (document.getElementById('alphagen-panel')?.value || '').trim();
            if (mode === 'multi' && panel) {
                await alphagenFetchMultiCandidates(panel);
            }
        });
    }

    const panelInput = document.getElementById('alphagen-panel');
    if (panelInput) {
        panelInput.addEventListener('change', async () => {
            const mode = (document.getElementById('alphagen-eval-mode')?.value || 'single');
            const panel = panelInput.value.trim();
            if (mode === 'multi' && panel) {
                await alphagenFetchMultiCandidates(panel);
            }
        });
    }

    const btnListExports = document.getElementById('btn-alphagen-list-exports');
    if (btnListExports) btnListExports.addEventListener('click', alphagenListExports);

    // Auto-load export list if possible
    alphagenListExports().catch(() => { });

    // Auto-load merged files list for convenience (delayed to ensure DOM ready)
    setTimeout(() => {
        alphagenLoadMergedFiles().catch(() => { });
    }, 500);

    // --- Onchain Dashboard Event Bindings ---
    const bindOnchain = (id, handler) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', handler);
    };

    bindOnchain('btn-onchain-flow', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-flow-params')?.value, 'Dune Params');
        await runOnchainAnalyze('exchange_flow', {
            ...common,
            exchange: document.getElementById('onchain-flow-exchange')?.value || 'binance',
            hours: Number(document.getElementById('onchain-flow-hours')?.value || 24),
            stablecoin: !!document.getElementById('onchain-flow-stablecoin')?.checked,
            dune_query: Number(document.getElementById('onchain-flow-dune')?.value || 0) || undefined,
            dune_params: duneParams
        }, 'onchain-flow-result');
    });

    bindOnchain('btn-onchain-behavior', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-behavior-params')?.value, 'Dune Params');
        await runOnchainAnalyze('address_behavior', {
            ...common,
            hours: Number(document.getElementById('onchain-behavior-hours')?.value || 24),
            retention_hours: Number(document.getElementById('onchain-behavior-retention')?.value || 24),
            addresses: document.getElementById('onchain-behavior-addresses')?.value || '',
            dune_query: Number(document.getElementById('onchain-behavior-dune')?.value || 0) || undefined,
            dune_params: duneParams
        }, 'onchain-behavior-result');
    });

    bindOnchain('btn-onchain-large', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-large-params')?.value, 'Dune Params');
        await runOnchainAnalyze('large_moves', {
            ...common,
            hours: Number(document.getElementById('onchain-large-hours')?.value || 24),
            min_value: Number(document.getElementById('onchain-large-min')?.value || 100),
            token: document.getElementById('onchain-large-token')?.value || '',
            contract_calls: !!document.getElementById('onchain-large-contract')?.checked,
            addresses: document.getElementById('onchain-large-addresses')?.value || '',
            dune_query: Number(document.getElementById('onchain-large-dune')?.value || 0) || undefined,
            dune_params: duneParams
        }, 'onchain-large-result');
    });

    bindOnchain('btn-onchain-mev', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-mev-params')?.value, 'Dune Params');
        await runOnchainAnalyze('mev', {
            ...common,
            dune_query: Number(document.getElementById('onchain-mev-dune')?.value || 0) || undefined,
            dune_params: duneParams
        }, 'onchain-mev-result');
    });

    bindOnchain('btn-onchain-gas', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-gas-params')?.value, 'Dune Params');
        await runOnchainAnalyze('gas', {
            ...common,
            hours: Number(document.getElementById('onchain-gas-hours')?.value || 1),
            max_blocks: Number(document.getElementById('onchain-gas-max-blocks')?.value || 200),
            dune_query: Number(document.getElementById('onchain-gas-dune')?.value || 0) || undefined,
            dune_params: duneParams
        }, 'onchain-gas-result');
    });

    bindOnchain('btn-onchain-protocol', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-protocol-params')?.value, 'Dune Params');
        const variables = parseJsonInput(document.getElementById('onchain-protocol-variables')?.value, 'Variables');
        await runOnchainAnalyze('protocol', {
            ...common,
            dune_query: Number(document.getElementById('onchain-protocol-dune')?.value || 0) || undefined,
            dune_params: duneParams,
            subgraph: document.getElementById('onchain-protocol-subgraph')?.value || '',
            query: document.getElementById('onchain-protocol-query')?.value || '',
            variables
        }, 'onchain-protocol-result');
    });

    bindOnchain('btn-onchain-cycle', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-cycle-params')?.value, 'Dune Params');
        await runOnchainAnalyze('capital_cycle', {
            ...common,
            hours: Number(document.getElementById('onchain-cycle-hours')?.value || 24),
            addresses: document.getElementById('onchain-cycle-addresses')?.value || '',
            dune_query: Number(document.getElementById('onchain-cycle-dune')?.value || 0) || undefined,
            dune_params: duneParams
        }, 'onchain-cycle-result');
    });

    bindOnchain('btn-onchain-token', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-token-params')?.value, 'Dune Params');
        await runOnchainAnalyze('token_distribution', {
            ...common,
            token: document.getElementById('onchain-token-token')?.value || '',
            whale_threshold: Number(document.getElementById('onchain-token-whale')?.value || 100000),
            top_n: Number(document.getElementById('onchain-token-topn')?.value || 10),
            addresses: document.getElementById('onchain-token-addresses')?.value || '',
            dune_query: Number(document.getElementById('onchain-token-dune')?.value || 0) || undefined,
            dune_params: duneParams
        }, 'onchain-token-result');
    });

    bindOnchain('btn-onchain-nft', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-nft-params')?.value, 'Dune Params');
        const variables = parseJsonInput(document.getElementById('onchain-nft-variables')?.value, 'Variables');
        await runOnchainAnalyze('nft', {
            ...common,
            dune_query: Number(document.getElementById('onchain-nft-dune')?.value || 0) || undefined,
            dune_params: duneParams,
            subgraph: document.getElementById('onchain-nft-subgraph')?.value || '',
            query: document.getElementById('onchain-nft-query')?.value || '',
            variables
        }, 'onchain-nft-result');
    });

    bindOnchain('btn-onchain-price', async () => {
        const common = getOnchainCommonPayload();
        const duneParams = parseJsonInput(document.getElementById('onchain-price-params')?.value, 'Dune Params');
        await runOnchainAnalyze('price_relation', {
            ...common,
            dune_query: Number(document.getElementById('onchain-price-dune')?.value || 0) || undefined,
            metric_key: document.getElementById('onchain-price-metric')?.value || '',
            exchange: document.getElementById('onchain-price-exchange')?.value || 'binance',
            symbol: document.getElementById('onchain-price-symbol')?.value || 'BTC/USDT',
            timeframe: document.getElementById('onchain-price-timeframe')?.value || '1h',
            limit: Number(document.getElementById('onchain-price-limit')?.value || 500),
            dune_params: duneParams
        }, 'onchain-price-result');
    });

    bindOnchain('btn-onchain-apply-template', () => {
        const key = document.getElementById('onchain-template-select')?.value;
        if (!key || !ONCHAIN_DUNE_TEMPLATES[key]) return;
        const tpl = ONCHAIN_DUNE_TEMPLATES[key];
        const paramsText = JSON.stringify(tpl.dune_params || {}, null, 2);

        const map = {
            exchange_flow: ['onchain-flow-params', 'onchain-flow-dune'],
            address_behavior: ['onchain-behavior-params', 'onchain-behavior-dune'],
            large_moves: ['onchain-large-params', 'onchain-large-dune'],
            mev: ['onchain-mev-params', 'onchain-mev-dune'],
            gas: ['onchain-gas-params', 'onchain-gas-dune'],
            protocol: ['onchain-protocol-params', 'onchain-protocol-dune'],
            capital_cycle: ['onchain-cycle-params', 'onchain-cycle-dune'],
            token_distribution: ['onchain-token-params', 'onchain-token-dune'],
            nft: ['onchain-nft-params', 'onchain-nft-dune'],
            price_relation: ['onchain-price-params', 'onchain-price-dune']
        };

        const target = map[key];
        if (!target) return;
        const paramsEl = document.getElementById(target[0]);
        const duneEl = document.getElementById(target[1]);
        if (paramsEl && !paramsEl.value.trim()) paramsEl.value = paramsText;
        if (duneEl && !duneEl.value.trim()) duneEl.placeholder = '填写你的 Query ID';
        if (key === 'price_relation') {
            const metricEl = document.getElementById('onchain-price-metric');
            if (metricEl && !metricEl.value.trim() && tpl.metric_key) metricEl.value = tpl.metric_key;
        }
        showAlert(`已应用模板：${key}`, 'success');
    });

    bindOnchain('btn-onchain-apply-subgraph', () => {
        const key = document.getElementById('onchain-subgraph-template')?.value;
        if (!key || !ONCHAIN_SUBGRAPH_TEMPLATES[key]) return;
        const tpl = ONCHAIN_SUBGRAPH_TEMPLATES[key];
        const subgraphEl = document.getElementById('onchain-protocol-subgraph');
        const queryEl = document.getElementById('onchain-protocol-query');
        const varsEl = document.getElementById('onchain-protocol-variables');
        if (subgraphEl) subgraphEl.value = tpl.subgraph || key;
        if (queryEl) queryEl.value = tpl.query || '';
        if (varsEl) varsEl.value = JSON.stringify(tpl.variables || {}, null, 2);
        showAlert('已应用子图模板', 'success');
    });

    bindOnchain('btn-onchain-apply-address', () => {
        const presetKey = document.getElementById('onchain-address-preset')?.value;
        const target = document.getElementById('onchain-address-target')?.value;
        if (!presetKey || !target) return;
        const list = ONCHAIN_ADDRESS_PRESETS[presetKey] || [];
        const text = list.join('\n');
        const idMap = {
            behavior: 'onchain-behavior-addresses',
            large: 'onchain-large-addresses',
            cycle: 'onchain-cycle-addresses',
            token: 'onchain-token-addresses'
        };
        const el = document.getElementById(idMap[target]);
        if (el) el.value = text;
        showAlert('已填入地址池', 'success');
    });

    const resultBtnMap = {
        exchange_flow: 'flow',
        address_behavior: 'behavior',
        large_moves: 'large',
        mev: 'mev',
        gas: 'gas',
        protocol: 'protocol',
        capital_cycle: 'cycle',
        token_distribution: 'token',
        nft: 'nft',
        price_relation: 'price'
    };

    Object.entries(resultBtnMap).forEach(([kind, suffix]) => {
        bindOnchain(`btn-onchain-${suffix}-copy`, () => copyOnchainResult(kind));
        bindOnchain(`btn-onchain-${suffix}-download`, () => downloadOnchainResult(kind));
        bindOnchain(`btn-onchain-${suffix}-csv`, () => downloadOnchainCsv(kind));
    });
});

async function loadAvailablePairs() {
    try {
        const data = await fetchJson('/api/system/trading_pairs', {}, { timeoutMs: 8000, retries: 1 });

        if (data.success && data.pairs) {
            const datalist = document.getElementById('available-pairs-list');
            if (datalist) {
                datalist.innerHTML = '';
                data.pairs.forEach(pair => {
                    const option = document.createElement('option');
                    option.value = pair;
                    datalist.appendChild(option);
                });
                console.log('Loaded ' + data.pairs.length + ' pairs for autocomplete');
            }
        }
    } catch (error) {
        console.error('Failed to load trading pairs:', error);
    }
}


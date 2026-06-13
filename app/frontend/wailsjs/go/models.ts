export namespace main {
	
	export class ArtifactRow {
	    path: string;
	    size: string;
	    sizeBytes: number;
	    modified: string;
	
	    static createFrom(source: any = {}) {
	        return new ArtifactRow(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.path = source["path"];
	        this.size = source["size"];
	        this.sizeBytes = source["sizeBytes"];
	        this.modified = source["modified"];
	    }
	}
	export class LiveStateResponse {
	    running: boolean;
	    current: string;
	    lastCode?: number;
	    snapshot?: Record<string, any>;
	
	    static createFrom(source: any = {}) {
	        return new LiveStateResponse(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.running = source["running"];
	        this.current = source["current"];
	        this.lastCode = source["lastCode"];
	        this.snapshot = source["snapshot"];
	    }
	}
	export class MetricRow {
	    step: number;
	    reward: number;
	    loss: number;
	    fps: number;
	    event: string;
	
	    static createFrom(source: any = {}) {
	        return new MetricRow(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.step = source["step"];
	        this.reward = source["reward"];
	        this.loss = source["loss"];
	        this.fps = source["fps"];
	        this.event = source["event"];
	    }
	}
	export class StateResponse {
	    repoRoot: string;
	    python: string;
	    pythonAvailable: boolean;
	    pythonError?: string;
	    missingTraining?: string[];
	    running: boolean;
	    current: string;
	    logs: string[];
	    lastCode?: number;
	    metrics: MetricRow[];
	    artifacts: ArtifactRow[];
	    snapshot?: Record<string, any>;
	
	    static createFrom(source: any = {}) {
	        return new StateResponse(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.repoRoot = source["repoRoot"];
	        this.python = source["python"];
	        this.pythonAvailable = source["pythonAvailable"];
	        this.pythonError = source["pythonError"];
	        this.missingTraining = source["missingTraining"];
	        this.running = source["running"];
	        this.current = source["current"];
	        this.logs = source["logs"];
	        this.lastCode = source["lastCode"];
	        this.metrics = this.convertValues(source["metrics"], MetricRow);
	        this.artifacts = this.convertValues(source["artifacts"], ArtifactRow);
	        this.snapshot = source["snapshot"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class TrainConfig {
	    timesteps: string;
	    modelOut: string;
	    resumeFrom: string;
	    device: string;
	    robotProfile: string;
	    preview: string;
	    nEnvs: string;
	    nSteps: string;
	    batchSize: string;
	    learningRate: string;
	    checkpointEvery: string;
	    pretrain: boolean;
	    advantageScope: boolean;
	
	    static createFrom(source: any = {}) {
	        return new TrainConfig(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.timesteps = source["timesteps"];
	        this.modelOut = source["modelOut"];
	        this.resumeFrom = source["resumeFrom"];
	        this.device = source["device"];
	        this.robotProfile = source["robotProfile"];
	        this.preview = source["preview"];
	        this.nEnvs = source["nEnvs"];
	        this.nSteps = source["nSteps"];
	        this.batchSize = source["batchSize"];
	        this.learningRate = source["learningRate"];
	        this.checkpointEvery = source["checkpointEvery"];
	        this.pretrain = source["pretrain"];
	        this.advantageScope = source["advantageScope"];
	    }
	}

}


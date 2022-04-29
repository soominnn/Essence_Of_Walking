
import './App.css';
import Unity, {UnityContext} from 'react-unity-webgl';

const unityContext = new UnityContext({
  loaderUrl: "Build/walk.loader.js",
  dataUrl: "Build/walk.data",
  frameworkUrl: "Build/walk.framework.js",
  codeUrl: "Build/walk.wasm",
});

function App() {
  return (
    <div className="App">
      <p>
        홍길동님의 결과
      </p>
      <Unity unityContext={unityContext} 
        style={{
          height: "200%",
          width: 800,
          border: "2px solid black",
          background: "grey",
        }}/>
    </div>
  );
}

export default App;

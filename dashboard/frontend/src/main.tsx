import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { DataSourceContext, dataSource } from "./data/DataSourceContext";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <DataSourceContext.Provider value={dataSource}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </DataSourceContext.Provider>
  </StrictMode>,
);

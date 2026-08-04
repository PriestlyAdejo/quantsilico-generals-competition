import { RouterProvider, createBrowserRouter } from "react-router";
import { routes } from "./routes";
import { DataSourceProvider } from "./DataSourceProvider";
import { Toaster } from "sonner";

const router = createBrowserRouter(routes);

export default function App() {
  return (
    <DataSourceProvider>
      <RouterProvider router={router} />
      <Toaster theme="dark" position="bottom-right" />
    </DataSourceProvider>
  );
}

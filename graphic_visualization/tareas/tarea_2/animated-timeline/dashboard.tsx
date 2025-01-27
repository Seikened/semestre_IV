import { Timeline } from "./components/timeline";

export default function Dashboard() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <div className="p-4 sm:p-8">
        <h1 className="text-3xl sm:text-4xl font-bold mb-4 text-center">
          Visualización interactiva
        </h1>
        <p className="text-muted-foreground mb-4 text-center">
          Cómo estas tecnologías han transformado la manera de interactuar con sistemas y datos.
        </p>
        <p className="text-muted-foreground text-center">
          Equipo 4: Fernando | Josué | Mares
        </p>
      </div>
      <div className="flex-grow">
        <Timeline />
      </div>
    </div>
  );
}
